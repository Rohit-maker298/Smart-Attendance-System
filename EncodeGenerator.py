import os
import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

print("=" * 60)
print("SMART ATTENDANCE - INSIGHTFACE ENCODING GENERATOR")
print("=" * 60)

# ----------------------------------------------------------------
# Model storage path: was hardcoded to 'D:/insightface_models',
# which breaks on any machine without that exact drive/folder.
# This resolves to a folder named 'insightface_models' next to
# this script, on whatever OS/drive it's run on. First run will
# download the buffalo_l model bundle into it (a few hundred MB).
# ----------------------------------------------------------------
MODELS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "insightface_models")

app = FaceAnalysis(name='buffalo_l', root=MODELS_ROOT, providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

folder_path = 'Images'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Created '{folder_path}' directory. Add student images and re-run.")
    exit()

known_embeddings = []
student_ids = []
valid_extensions = ('.jpg', '.png', '.jpeg')
processed_count = 0
skipped = []

for path in sorted(os.listdir(folder_path)):
    if not path.lower().endswith(valid_extensions):
        continue

    img_path = os.path.join(folder_path, path)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[SKIP] Could not read image: {path}")
        skipped.append(path)
        continue

    faces = app.get(img)

    if len(faces) == 0:
        print(f"[SKIP] No face detected in: {path}")
        skipped.append(path)
        continue

    if len(faces) > 1:
        # Enrollment photos should have exactly one person. If more than one
        # face is found, the largest bounding box is almost always the
        # intended subject (others are usually background people).
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        print(f"[WARNING] {len(faces)} faces found in {path} — using the largest one. "
              f"Consider re-cropping this photo to just the student.")

    face = faces[0]
    embedding = face.embedding
    embedding = embedding / np.linalg.norm(embedding)  # normalize for cosine similarity later

    known_embeddings.append(embedding)
    student_id = os.path.splitext(path)[0]
    student_ids.append(student_id)
    processed_count += 1
    print(f"[SUCCESS] Encoded (ArcFace): {student_id}")

if known_embeddings:
    with open("EFile.p", "wb") as f:
        pickle.dump([known_embeddings, student_ids], f)
    print(f"\nSUCCESS: Encoded {processed_count} students into 'EFile.p'.")
    if skipped:
        print(f"Skipped {len(skipped)} file(s): {', '.join(skipped)}")
else:
    print("\nNo faces were encoded. Check that 'Images/' contains valid, clear face photos.")