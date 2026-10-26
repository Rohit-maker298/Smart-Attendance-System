import os
import io
import cv2
import time
import pickle
import base64
import threading
import numpy as np
from datetime import datetime
from collections import deque
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
from flask import send_from_directory


# Deep learning vision engine — identity matching only
from insightface.app import FaceAnalysis

# Lightweight landmark model — liveness (blink) signal only.
# Kept separate from InsightFace deliberately: its eye landmark indices
# are a scheme we've already verified, whereas InsightFace's 106-point
# layout uses a different, less-documented ordering. Getting liveness
# landmark indices wrong would silently produce a liveness check that
# looks like it works but doesn't actually catch anything — worse than
# not having one, so this uses the known-correct indices instead.
import mediapipe as mp

# Cloud storage / database
import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)

# ==========================================
# 1. CONFIG
# ==========================================
MODELS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "insightface_models")
FIREBASE_KEY_PATH = "serviceAccountKey.json"

# TODO: replace with your actual Firebase Realtime Database URL —
# find it in Firebase console > Realtime Database > top of the page.
# It looks like: https://<project-id>-default-rtdb.<region>.firebasedatabase.app/
FIREBASE_DATABASE_URL = "https://YOUR_FIREBASE_PROJECT_ID-default-rtdb.firebaseio.com/"

SIMILARITY_THRESHOLD = 0.45        # ArcFace cosine similarity — starting point, calibrate on your own enrollment photos
BLINK_EAR_THRESHOLD = 0.21         # eye-aspect-ratio below this = eyes considered closed — starting point, calibrate per webcam/lighting
LIVENESS_WINDOW_SECONDS = 8        # how long a matched student has to blink before the match resets
PROCESS_WIDTH = 480                # frames wider than this get downscaled before InsightFace runs, for speed

# ==========================================
# 2. FIREBASE INITIALIZATION
# ==========================================
firebase_enabled = False

if "YOUR_FIREBASE_PROJECT_ID" in FIREBASE_DATABASE_URL:
    print("[FIREBASE] Not configured — FIREBASE_DATABASE_URL is still the placeholder. "
          "Attendance will be recorded locally only. Update the URL in app.py to enable cloud sync.")
elif not os.path.exists(FIREBASE_KEY_PATH):
    print(f"[FIREBASE] '{FIREBASE_KEY_PATH}' not found. Running in local-only mode.")
else:
    try:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DATABASE_URL})
        firebase_enabled = True
        print("[FIREBASE] Connected to Realtime Database.")
    except Exception as e:
        print(f"[FIREBASE] Initialization failed: {e}")

def sync_to_firebase(student_id, record):
    """Runs in a background thread — must never block the request path."""
    if not firebase_enabled:
        return
    try:
        today_date = record['date']
        ref = db.reference(f'Attendance/{today_date}/{student_id}')
        ref.set({
            'time': record['time'],
            'confidence': record['confidence'],
            'status': 'Present'
        })
        print(f"[FIREBASE SYNC] Marked {student_id} under {today_date}")
    except Exception as e:
        print(f"[FIREBASE ERROR] Failed to sync {student_id}: {e}")

# ==========================================
# 3. INSIGHTFACE (ARCFACE) — IDENTITY
# ==========================================
print("Loading InsightFace ArcFace engine...")
face_app = FaceAnalysis(name='buffalo_l', root=MODELS_ROOT, providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace ready.")

known_vectors = np.empty((0, 512), dtype='float32')
student_ids = []

def load_encodings():
    global known_vectors, student_ids
    if os.path.exists("EFile.p"):
        try:
            with open("EFile.p", "rb") as f:
                vectors_list, student_ids = pickle.load(f)
            if len(vectors_list) > 0:
                known_vectors = np.array(vectors_list, dtype='float32')
                print(f"Loaded {len(student_ids)} student profiles from 'EFile.p'.")
        except Exception as e:
            print(f"[LOAD ERROR] {e}")
    else:
        print("'EFile.p' not found! Run EncodeGenerator.py first.")

load_encodings()

def find_best_match(live_vector):
    """Vectorized cosine similarity against all known ArcFace embeddings."""
    if known_vectors.shape[0] == 0:
        return None, 0.0
    live_vector = live_vector / np.linalg.norm(live_vector)
    scores = np.dot(known_vectors, live_vector)
    best_idx = np.argmax(scores)
    best_score = float(scores[best_idx])
    if best_score >= SIMILARITY_THRESHOLD:
        return student_ids[best_idx], best_score
    return None, best_score

# ==========================================
# 4. MEDIAPIPE FACE MESH — LIVENESS ONLY
# ==========================================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Verified eye landmark indices for MediaPipe FaceMesh (468-point model)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(points, eye_idx):
    p1, p2, p3, p4, p5, p6 = [points[i] for i in eye_idx]
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return None
    return vertical / horizontal

def get_ear_for_crop(face_crop_rgb):
    """Runs FaceMesh on just the matched student's cropped region and returns
    average eye-aspect-ratio, or None if no landmarks could be extracted."""
    if face_crop_rgb.size == 0:
        return None
    res = mp_face_mesh.process(face_crop_rgb)
    if not res.multi_face_landmarks:
        return None
    h, w = face_crop_rgb.shape[:2]
    landmarks = res.multi_face_landmarks[0].landmark
    points = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype='float32')
    right_ear = eye_aspect_ratio(points, RIGHT_EYE)
    left_ear = eye_aspect_ratio(points, LEFT_EYE)
    if right_ear is None or left_ear is None:
        return None
    return (right_ear + left_ear) / 2.0

# ==========================================
# 5. STATE
# ==========================================
attendance_log = {}          # {student_id: {'time':..., 'date':..., 'confidence':...}}
pending_liveness = {}        # {student_id: {'history': deque, 'saw_closed': bool, 'start': ts}}

def update_liveness(student_id, ear):
    """
    Blink state machine: waits for the eye-aspect-ratio to dip below the
    closed-eye threshold and then recover above it — that open->closed->open
    sequence is a blink. A static photo held up to the camera holds one EAR
    value indefinitely and will never produce this transition, which is
    exactly the signal we're using to reject it.

    Returns one of: 'awaiting_blink', 'verified', 'timeout'
    """
    now = time.time()
    state = pending_liveness.get(student_id)

    if state is None:
        state = {'saw_closed': False, 'start': now}
        pending_liveness[student_id] = state

    if now - state['start'] > LIVENESS_WINDOW_SECONDS:
        pending_liveness.pop(student_id, None)
        return 'timeout'

    if ear is not None:
        if ear < BLINK_EAR_THRESHOLD:
            state['saw_closed'] = True
        elif state['saw_closed'] and ear >= BLINK_EAR_THRESHOLD:
            pending_liveness.pop(student_id, None)
            return 'verified'

    return 'awaiting_blink'

# ==========================================
# 6. PAGE ROUTES
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

# Add near the top with other routes
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS)"""
    return send_from_directory('static', filename)

# ==========================================
# 7. API ENDPOINTS
# ==========================================
@app.route('/get_attendance')
def get_attendance():
    return jsonify(attendance_log)

@app.route('/student_image/<student_id>')
def get_student_image(student_id):
    """Serves the enrollment photo for the visual confirmation card."""
    folder_path = 'Images'
    # Try common image extensions
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        image_path = os.path.join(folder_path, f"{student_id}{ext}")
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
    return jsonify({'error': 'Image not found'}), 404

@app.route('/export')
def export_excel():
    if not attendance_log:
        df = pd.DataFrame(columns=["Student ID", "Time", "Date", "Confidence", "Status"])
    else:
        rows = [
            {
                "Student ID": name,
                "Time": rec['time'],
                "Date": rec['date'],
                "Confidence": f"{rec['confidence']}%",
                "Status": "Present"
            }
            for name, rec in attendance_log.items()
        ]
        df = pd.DataFrame(rows)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Attendance')
    output.seek(0)

    filename = f"Attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return send_file(
        output,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name=filename
    )

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Downscale before running detection — real latency win, same principle
        # as your MediaPipe-only version.
        h0, w0 = frame.shape[:2]
        if w0 > PROCESS_WIDTH:
            scale = PROCESS_WIDTH / w0
            frame = cv2.resize(frame, (PROCESS_WIDTH, int(h0 * scale)), interpolation=cv2.INTER_AREA)

        detected_faces = face_app.get(frame)
        annotated_frame = frame.copy()
        latest_matches = []
        detections = []

        for face in detected_faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
            face_crop_bgr = frame[y1:y2, x1:x2]

            name, score = find_best_match(face.embedding)

            if name is None:
                box_color = (128, 128, 128)  # gray — unrecognized, not a spoof judgment
                label = "Unknown"
                status = "unknown"

            elif name in attendance_log:
                box_color = (0, 200, 0)
                label = f"{name} - Present"
                status = "already_marked"

            else:
                face_crop_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                ear = get_ear_for_crop(face_crop_rgb)
                liveness_status = update_liveness(name, ear)

                if liveness_status == 'verified':
                    now = datetime.now()
                    record = {
                        'time': now.strftime("%H:%M:%S"),
                        'date': now.strftime("%Y-%m-%d"),
                        'confidence': round(score * 100, 1)
                    }
                    attendance_log[name] = record
                    latest_matches.append(name)
                    threading.Thread(target=sync_to_firebase, args=(name, record), daemon=True).start()

                    box_color = (0, 255, 0)  # green — verified
                    label = f"{name} ({int(score * 100)}%) - Verified"
                    status = "verified"

                elif liveness_status == 'timeout':
                    box_color = (0, 0, 255)  # red — didn't blink in time
                    label = f"{name} - Blink timeout, retry"
                    status = "timeout"

                else:
                    box_color = (0, 165, 255)  # amber — matched, waiting on blink
                    label = f"{name} ({int(score * 100)}%) - Blink to verify"
                    status = "awaiting_blink"

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.rectangle(annotated_frame, (x1, max(0, y1 - 22)), (x1 + 220, y1), box_color, -1)
            cv2.putText(annotated_frame, label, (x1 + 5, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            detections.append({'name': name if name else 'unknown', 'score': round(score * 100, 1), 'status': status})

        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        annotated_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'logs': attendance_log,
            'latest_matches': latest_matches,
            'detections': detections,
            'annotated_image': annotated_image,
            'face_count': len(detected_faces)
        })

    except Exception as e:
        print(f"[ERROR] Frame processing failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=False, port=5000)