// ==========================================================
// PRESENCE — Attendance / scan page
//
// The live <video> element is always playing at full camera frame
// rate — it is never blocked on the network. Recognition frames are
// captured and posted to /process_frame in the background; the
// response only updates a small confirmation thumbnail, the scan
// frame's status color, and the banner/log. This keeps the picture
// smooth even while the recognition model is still thinking.
// ==========================================================

const CAPTURE_WIDTH = 420;      // frame is downscaled before sending — smaller payload, faster round trip
const JPEG_QUALITY = 0.6;
const MIN_GAP_MS = 250;         // shortest pause between the end of one request and the start of the next
const THUMB_HOLD_MS = 4000;     // how long the "last match" thumbnail stays visible after a detection

const video = document.getElementById('video');
const placeholder = document.getElementById('cameraPlaceholder');
const toggleBtn = document.getElementById('toggleBtn');
const statusDot = document.getElementById('statusDot');
const statusLabel = document.getElementById('statusLabel');
const faceCountEl = document.getElementById('faceCount');
const resultBanner = document.getElementById('resultBanner');
const resultText = document.getElementById('resultText');
const sessionLog = document.getElementById('sessionLog');
const logCount = document.getElementById('logCount');
const clockEl = document.getElementById('clock');
const scanFrame = document.getElementById('scanFrame');
const captureThumb = document.getElementById('captureThumb');
const thumbImg = document.getElementById('thumbImg');

const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d');

let stream = null;
let scanning = false;
let loopHandle = null;
let thumbHideTimer = null;
const loggedThisSession = new Set();

// ---------- clock ----------
function tickClock() {
  clockEl.textContent = new Date().toLocaleTimeString('en-GB');
}
tickClock();
setInterval(tickClock, 1000);

// ---------- status banner + frame color ----------
const STATUS_PRIORITY = ['verified', 'timeout', 'awaiting_blink', 'already_marked', 'unknown'];

const STATUS_COPY = {
  verified: (d) => ['state-verified', `${d.name} verified — attendance marked.`],
  timeout: (d) => ['state-danger', `${d.name} matched but didn't blink in time — hold still and try again.`],
  awaiting_blink: (d) => ['state-amber', `${d.name} matched — blink to confirm you're really there.`],
  already_marked: (d) => ['state-verified', `${d.name} is already marked present today.`],
  unknown: () => [null, `Face detected but not recognized. Make sure enrollment photos are up to date.`],
};

function clearFrameState() {
  scanFrame.classList.remove('state-verified', 'state-amber', 'state-danger');
}

function updateBanner(detections) {
  resultBanner.classList.remove('verified', 'amber', 'danger');
  clearFrameState();

  if (!detections || detections.length === 0) {
    resultText.textContent = 'Looking for a face…';
    return;
  }

  let chosen = null;
  for (const status of STATUS_PRIORITY) {
    chosen = detections.find((d) => d.status === status);
    if (chosen) break;
  }
  if (!chosen) chosen = detections[0];

  const builder = STATUS_COPY[chosen.status];
  const [stateClass, text] = builder ? builder(chosen) : [null, 'Processing…'];

  if (stateClass) {
    scanFrame.classList.add(stateClass);
    resultBanner.classList.add(
      stateClass === 'state-verified' ? 'verified' : stateClass === 'state-amber' ? 'amber' : 'danger'
    );
  }
  resultText.textContent = text;
}

// ---------- confirmation thumbnail ----------
function showThumb(base64Jpg) {
  thumbImg.src = `data:image/jpeg;base64,${base64Jpg}`;
  captureThumb.classList.add('show');
  clearTimeout(thumbHideTimer);
  thumbHideTimer = setTimeout(() => captureThumb.classList.remove('show'), THUMB_HOLD_MS);
}

// ---------- session log ----------
function addToLog(name) {
  if (loggedThisSession.has(name)) return;
  loggedThisSession.add(name);

  const empty = sessionLog.querySelector('.log-empty');
  if (empty) empty.remove();

  const item = document.createElement('div');
  item.className = 'log-item';
  item.innerHTML = `
    <img src="${studentPhotoUrl(name)}" alt="" onerror="this.src='${initialsAvatar(name)}'">
    <div class="meta">
      <div class="n">${name}</div>
      <div class="t">${new Date().toLocaleTimeString('en-GB')}</div>
    </div>
    <span class="badge badge-verified">Verified</span>
  `;
  sessionLog.prepend(item);
  logCount.textContent = `${loggedThisSession.size} today`;
}

// ---------- camera lifecycle ----------
async function startScanning() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
    });
  } catch (err) {
    resultBanner.classList.add('danger');
    resultText.textContent = 'Camera access was denied or is unavailable. Allow camera permissions and try again.';
    return;
  }

  video.srcObject = stream;
  await video.play();

  placeholder.style.display = 'none';

  scanning = true;
  toggleBtn.textContent = 'Stop Scanning';
  toggleBtn.classList.add('btn-danger');
  statusDot.classList.add('live');
  statusLabel.textContent = 'Scanning';
  resultText.textContent = 'Looking for a face…';

  loop();
}

function stopScanning() {
  scanning = false;
  clearTimeout(loopHandle);
  loopHandle = null;

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }

  toggleBtn.textContent = 'Start Scanning';
  toggleBtn.classList.remove('btn-danger');
  statusDot.classList.remove('live');
  statusLabel.textContent = 'Camera idle';
  faceCountEl.textContent = '';
  resultBanner.classList.remove('verified', 'amber', 'danger');
  clearFrameState();
  captureThumb.classList.remove('show');
  resultText.textContent = 'Start scanning to begin marking attendance.';

  placeholder.style.display = 'flex';
}

toggleBtn.addEventListener('click', () => {
  if (scanning) stopScanning();
  else startScanning();
});

// ---------- capture + send loop ----------
// Self-scheduling instead of a fixed setInterval: the next capture only
// fires MIN_GAP_MS after the previous request finishes, so a slow response
// from the recognition model just means a slightly longer gap — it never
// piles up queued requests or stalls the visible video.
function loop() {
  if (!scanning) return;
  captureAndSend().finally(() => {
    if (scanning) loopHandle = setTimeout(loop, MIN_GAP_MS);
  });
}

async function captureAndSend() {
  if (!video.videoWidth) return;

  const scale = Math.min(1, CAPTURE_WIDTH / video.videoWidth);
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY);

  try {
    const res = await fetch('/process_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    faceCountEl.textContent = data.face_count > 0
      ? `${data.face_count} face${data.face_count > 1 ? 's' : ''} detected`
      : 'No face detected';

    updateBanner(data.detections);

    const hasNamedFace = (data.detections || []).some((d) => d.name && d.name !== 'unknown');
    if (data.annotated_image && hasNamedFace) showThumb(data.annotated_image);

    (data.latest_matches || []).forEach(addToLog);
  } catch (err) {
    console.error('process_frame failed:', err);
  }
}

// Seed the session log from whatever is already recorded today, so a page
// refresh mid-class doesn't lose the list.
(async function seedLogFromServer() {
  try {
    const res = await fetch('/get_attendance');
    if (!res.ok) return;
    const log = await res.json();
    const entries = Object.entries(log).sort((a, b) => (a[1].time > b[1].time ? -1 : 1));
    if (entries.length === 0) return;

    const empty = sessionLog.querySelector('.log-empty');
    if (empty) empty.remove();

    entries.forEach(([name, rec]) => {
      loggedThisSession.add(name);
      const item = document.createElement('div');
      item.className = 'log-item';
      item.innerHTML = `
        <img src="${studentPhotoUrl(name)}" alt="" onerror="this.src='${initialsAvatar(name)}'">
        <div class="meta">
          <div class="n">${name}</div>
          <div class="t">${rec.time}</div>
        </div>
        <span class="badge badge-verified">Verified</span>
      `;
      sessionLog.appendChild(item);
    });
    logCount.textContent = `${loggedThisSession.size} today`;
  } catch (err) {
    // Non-fatal — the log will just start empty.
  }
})();
