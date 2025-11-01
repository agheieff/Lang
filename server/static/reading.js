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

// Lightweight readiness poller for enabling the Next button
(function () {
  async function pollNextReady() {
    const btn = document.getElementById('next-btn');
    const status = document.getElementById('next-status');
    if (!btn) return;
    try {
      const wait = (window.NEXT_READY_MAX_WAIT_SEC || 8);
      const r = await fetch(`/reading/next/ready?wait=${encodeURIComponent(String(wait))}`, {
        headers: { 'Accept': 'application/json' }
      });
      const data = await r.json().catch(() => null);
      if (data && data.ready) {
        btn.disabled = false;
        btn.removeAttribute('aria-disabled');
        if (status) status.textContent = '';
        return; // stop polling when ready
      }
    } catch (e) {}
    setTimeout(pollNextReady, 1000);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', pollNextReady);
  } else {
    pollNextReady();
  }
})();
