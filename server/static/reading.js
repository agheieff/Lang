// Minimal client helpers
window.arcToggleTranslation = function (e) {
  const btn = e && e.target;
  const panel = document.getElementById('translation-panel');
  if (panel) {
    const show = panel.classList.contains('hidden');
    panel.hidden = !show;
    panel.classList.toggle('hidden', !show);
    if (btn) btn.setAttribute('aria-expanded', String(show));
  }
};

window.arcBuildNextParams = function () {
  return { session_id: null, read_time_ms: null };
};

// NOTE: Next button readiness polling is handled per-page (e.g., in home.html template)
// This avoids duplicate polling requests that previously occurred with the global poller
