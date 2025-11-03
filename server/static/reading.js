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
  // Get the current session data from local storage
  const textEl = document.getElementById('reading-text');
  if (!textEl || !textEl.dataset.textId) {
    return { session_id: null, read_time_ms: null };
  }
  
  const textId = textEl.dataset.textId;
  const sessionKey = localStorage.getItem(`arc_current_session_${textId}`);
  if (!sessionKey) {
    return { session_id: null, read_time_ms: null };
  }
  
  try {
    const sessionData = JSON.parse(localStorage.getItem(sessionKey) || '{}');
    const now = Date.now();
    
    // Update analytics before sending
    if (sessionData.analytics) {
      const readingTime = now - (sessionData.opened_at || now);
      sessionData.analytics.reading_time_ms = readingTime;
      if (sessionData.analytics.total_words > 0) {
        sessionData.analytics.average_reading_speed_wpm = Math.round((sessionData.analytics.total_words / readingTime) * 60000);
      }
      sessionData.analytics.completion_status = 'finished';
    }
    
    // Save updated session
    try { localStorage.setItem(sessionKey, JSON.stringify(sessionData)); } catch {}
    
    console.log('Sending session data to server for next text:', sessionData);
    console.log('Session data keys:', Object.keys(sessionData || {}));
    console.log('Text ID:', sessionData?.text_id);
    console.log('Session ID:', sessionData?.session_id);
    
    // Store session data for the form to pick up
    try { sessionStorage.setItem('arc_next_params', JSON.stringify(sessionData)); } catch {}
    
    return sessionData;
  } catch (e) {
    console.error('Error building next params:', e);
    return { session_id: null, read_time_ms: null };
  }
};

// Add function to handle the next button click properly
window.handleNextText = function() {
  const params = window.arcBuildNextParams();
  console.log('Next button clicked, sending:', params);
  
  // Create and submit a form with the session data
  const form = document.createElement('form');
  form.method = 'POST';
  form.action = '/reading/next';
  form.style.display = 'none';
  
  const hiddenField = document.createElement('input');
  hiddenField.type = 'hidden';
  hiddenField.name = 'session_data';
  hiddenField.value = JSON.stringify(params);
  form.appendChild(hiddenField);
  
  document.body.appendChild(form);
  form.submit();
};

// NOTE: Next button readiness polling is handled per-page (e.g., in home.html template)
// This avoids duplicate polling requests that previously occurred with the global poller
