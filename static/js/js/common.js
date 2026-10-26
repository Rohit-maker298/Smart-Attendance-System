// Shared across all three pages — highlights the current page in the nav.
(function () {
  const path = window.location.pathname;
  document.querySelectorAll('#navLinks a').forEach((link) => {
    const href = link.getAttribute('href');
    const isMatch = href === path || (href !== '/' && path.startsWith(href));
    if (isMatch) link.classList.add('active');
  });
})();

// Small helper used by dashboard.js / attendance.js to resolve a student's
// enrollment photo via the /student_image/<id> endpoint, with a graceful
// fallback if the photo 404s.
function studentPhotoUrl(studentId) {
  return `/student_image/${encodeURIComponent(studentId)}`;
}

function initialsAvatar(name) {
  const initials = (name || '?').slice(0, 2).toUpperCase();
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">
    <rect width="100%" height="100%" fill="#16213a"/>
    <text x="50%" y="54%" fill="#45e8d4" font-family="Space Grotesk, sans-serif"
      font-size="14" text-anchor="middle">${initials}</text>
  </svg>`;
  return `data:image/svg+xml;base64,${btoa(svg)}`;
}