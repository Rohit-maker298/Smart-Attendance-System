// ==========================================================
// PRESENCE — Dashboard
// Polls the existing /get_attendance endpoint (no backend
// changes needed) and renders stats, the attendance ring,
// and the live check-in table.
// ==========================================================

// TODO: set this to your actual class/roster size — same pattern as the
// FIREBASE_DATABASE_URL placeholder in app.py. There's no roster endpoint
// yet, so this is the one number the UI can't infer on its own.
const TOTAL_STUDENTS = 40;

const POLL_INTERVAL_MS = 3000;

const els = {
  statPresent: document.getElementById('statPresent'),
  statPresentSub: document.getElementById('statPresentSub'),
  statConfidence: document.getElementById('statConfidence'),
  statLastTime: document.getElementById('statLastTime'),
  statLastName: document.getElementById('statLastName'),
  statTotal: document.getElementById('statTotal'),
  ringFill: document.getElementById('ringFill'),
  ringNumber: document.getElementById('ringNumber'),
  ringLabel: document.getElementById('ringLabel'),
  rosterTable: document.getElementById('rosterTable'),
  rosterBody: document.getElementById('rosterBody'),
  emptyState: document.getElementById('emptyState'),
  searchInput: document.getElementById('searchInput'),
};

const RING_RADIUS = 86;
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;
els.ringFill.setAttribute('stroke-dasharray', RING_CIRCUMFERENCE.toFixed(1));

els.statTotal.textContent = TOTAL_STUDENTS > 0 ? TOTAL_STUDENTS : '—';

let latestLog = {};

function setRing(present, total) {
  const pct = total > 0 ? Math.min(present / total, 1) : 0;
  const offset = RING_CIRCUMFERENCE * (1 - pct);
  els.ringFill.style.strokeDashoffset = offset.toFixed(1);
  els.ringNumber.textContent = present;
  els.ringLabel.textContent = total > 0 ? `of ${total} present` : 'present';
}

function renderStats(log) {
  const entries = Object.entries(log);
  const count = entries.length;

  els.statPresent.textContent = count;
  els.statPresentSub.textContent = count === 0
    ? 'No check-ins yet'
    : `${Math.max(TOTAL_STUDENTS - count, 0)} not yet marked`;

  if (count === 0) {
    els.statConfidence.textContent = '—';
    els.statLastTime.textContent = '—';
    els.statLastName.textContent = 'Waiting for first scan';
  } else {
    const avgConf = entries.reduce((sum, [, r]) => sum + Number(r.confidence || 0), 0) / count;
    els.statConfidence.textContent = `${avgConf.toFixed(1)}%`;

    let last = entries[0];
    for (const e of entries) {
      const key = `${e[1].date}T${e[1].time}`;
      const lastKey = `${last[1].date}T${last[1].time}`;
      if (key > lastKey) last = e;
    }
    els.statLastTime.textContent = last[1].time;
    els.statLastName.textContent = last[0];
  }

  setRing(count, TOTAL_STUDENTS);
}

function renderTable(log, filterText) {
  const entries = Object.entries(log).sort((a, b) => {
    const ka = `${a[1].date}T${a[1].time}`;
    const kb = `${b[1].date}T${b[1].time}`;
    return kb.localeCompare(ka); // newest first
  });

  const filtered = filterText
    ? entries.filter(([id]) => id.toLowerCase().includes(filterText.toLowerCase()))
    : entries;

  if (entries.length === 0) {
    els.rosterTable.style.display = 'none';
    els.emptyState.style.display = 'block';
    return;
  }

  els.rosterTable.style.display = 'table';
  els.emptyState.style.display = 'none';

  els.rosterBody.innerHTML = filtered.map(([id, rec]) => `
    <tr>
      <td>
        <div class="who">
          <img class="avatar" src="${studentPhotoUrl(id)}" alt=""
               onerror="this.src='${initialsAvatar(id)}'">
          <div>
            <div class="name">${escapeHtml(id)}</div>
            <div class="id">ID · ${escapeHtml(id)}</div>
          </div>
        </div>
      </td>
      <td class="time">${rec.time}</td>
      <td class="time">${rec.date}</td>
      <td class="conf">${Number(rec.confidence).toFixed(1)}%</td>
      <td><span class="badge badge-verified">Verified</span></td>
    </tr>
  `).join('');

  if (filtered.length === 0) {
    els.rosterBody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--muted-2);padding:24px;">No students match "${escapeHtml(filterText)}"</td></tr>`;
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

async function refresh() {
  try {
    const res = await fetch('/get_attendance');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    latestLog = await res.json();
    renderStats(latestLog);
    renderTable(latestLog, els.searchInput.value.trim());
  } catch (err) {
    console.error('Failed to refresh attendance:', err);
  }
}

els.searchInput.addEventListener('input', () => {
  renderTable(latestLog, els.searchInput.value.trim());
});

refresh();
setInterval(refresh, POLL_INTERVAL_MS);