// Minimal client helpers
window.arcToggleTranslation = function (e) {
  const btn = e && e.target;
  const panel = document.getElementById('translation-panel');
  if (!panel) return;

  const show = panel.classList.contains('hidden');
  panel.hidden = !show;
  panel.classList.toggle('hidden', !show);
  if (btn) btn.setAttribute('aria-expanded', String(show));

  // When opening the panel, show a placeholder until translations arrive
  if (show) {
    const textEl = document.getElementById('translation-text');
    if (textEl && !textEl.textContent.trim()) {
      textEl.textContent = 'Translation is not ready yet.';
    }
  }
};

// Extract lookups and interactions from nested session structure
function extractWordEvents(sessionData) {
  const lookups = [];
  const interactions = [];
  const exposures = [];
  
  // Helper to create interaction/lookup entry
  function addWordEvent(word, isLookup) {
    const entry = {
      word: word.surface,
      lemma: word.lemma || word.surface,
      pos: word.pos || null,
      span_start: word.span_start,
      span_end: word.span_end,
      translation: word.translation || null,
      pinyin: word.pinyin || null
    };
    
    if (isLookup && word.looked_up_at) {
      // Word was clicked/looked up
      lookups.push({
        ...entry,
        timestamp: new Date(word.looked_up_at).toISOString(),
        translations: word.translation ? [word.translation] : []
      });
      interactions.push({
        ...entry,
        event_type: 'click',
        timestamp: new Date(word.looked_up_at).toISOString()
      });
    }
    
    // All words are exposures
    exposures.push({
      ...entry,
      event_type: 'exposure',
      timestamp: new Date(sessionData.opened_at || Date.now()).toISOString()
    });
  }
  
  // Process title words
  if (sessionData.title && Array.isArray(sessionData.title.words)) {
    for (const word of sessionData.title.words) {
      addWordEvent(word, !!word.looked_up_at);
    }
  }
  
  // Process paragraph/sentence words
  if (Array.isArray(sessionData.paragraphs)) {
    for (const para of sessionData.paragraphs) {
      if (Array.isArray(para.sentences)) {
        for (const sentence of para.sentences) {
          if (Array.isArray(sentence.words)) {
            for (const word of sentence.words) {
              addWordEvent(word, !!word.looked_up_at);
            }
          }
        }
      }
    }
  }
  
  return { lookups, interactions, exposures };
}

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
    
    // Extract word events in server-expected format
    const { lookups, interactions, exposures } = extractWordEvents(sessionData);
    sessionData.lookups = lookups;
    sessionData.interactions = [...interactions, ...exposures];
    
    // Save updated session
    try { localStorage.setItem(sessionKey, JSON.stringify(sessionData)); } catch {}
    
    console.log('Sending session data to server:', {
      text_id: sessionData.text_id,
      lookups_count: lookups.length,
      interactions_count: sessionData.interactions.length
    });
    
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
