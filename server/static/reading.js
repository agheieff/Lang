// Minimal client helpers
window.arcToggleTranslation = async function (e) {
  const btn = e && e.target;
  const panel = document.getElementById('translation-panel');
  if (!panel) return;

  const show = panel.classList.contains('hidden');
  panel.hidden = !show;
  panel.classList.toggle('hidden', !show);
  if (btn) btn.setAttribute('aria-expanded', String(show));

  // When opening the panel, fetch translation if missing
  if (show) {
    const textEl = document.getElementById('translation-text');
    if (textEl && !textEl.textContent.trim()) {
      textEl.textContent = 'Loading translation...';
      
      // Fetch full text translation
      const mainTextEl = document.getElementById('reading-text');
      const textId = mainTextEl ? mainTextEl.dataset.textId : null;
      
      if (textId) {
        try {
          const res = await fetch(`/reading/${textId}/translations?unit=text`, { 
            headers: { 'Accept': 'application/json' } 
          });
          
          if (res.ok) {
            const data = await res.json();
            if (data.items && data.items.length > 0) {
              // Find the longest translation (likely the body, not title)
              // or sort by length descending
              const items = data.items.sort((a, b) => (b.translation?.length || 0) - (a.translation?.length || 0));
              const best = items[0];
              
              if (best && best.translation) {
                textEl.textContent = best.translation;
                
                // Cache it in session data for persistence
                const sessionKey = localStorage.getItem(`arc_current_session_${textId}`);
                if (sessionKey) {
                  const sessionData = JSON.parse(localStorage.getItem(sessionKey) || '{}');
                  sessionData.full_translation = best.translation;
                  localStorage.setItem(sessionKey, JSON.stringify(sessionData));
                }
                return;
              }
            }
          }
          textEl.textContent = 'Translation is not ready yet.';
        } catch (err) {
          console.error('Failed to fetch translation:', err);
          textEl.textContent = 'Error loading translation.';
        }
      } else {
        textEl.textContent = 'Translation is not ready yet.';
      }
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

window.arcSyncSession = function() {
  const sessionData = window.arcBuildNextParams();
  if (!sessionData || !sessionData.text_id) return;
  
  // Only sync if we have actual data
  const keys = Object.keys(sessionData);
  if (keys.length <= 2 && !sessionData.clicks?.length) return;

  fetch('/reading/sync', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(sessionData)
  }).catch(err => console.warn('Background sync failed:', err));
};

// Set up periodic sync
setInterval(window.arcSyncSession, 30000); // Every 30s

// Add function to handle the next button click properly
window.handleNextText = function() {
  window.arcSyncSession(); // Final sync
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

window.arcRestoreSession = function() {
  const stateEl = document.getElementById('reading-session-state');
  if (!stateEl) return;
  
  try {
    const serverState = JSON.parse(stateEl.textContent);
    if (!serverState || !serverState.text_id) return;
    
    const textEl = document.getElementById('reading-text');
    if (!textEl || textEl.dataset.textId != serverState.text_id) return;
    
    console.log('Restoring session from server state:', serverState);
    
    // Merge with local storage if needed, or just use server state
    // Ideally we prefer server state as "truth" on reload, but local might be fresher if offline
    // For now, let's update local storage with server state if local is empty/older
    const sessionKey = `arc_current_session_${serverState.text_id}`;
    const localState = JSON.parse(localStorage.getItem(sessionKey) || '{}');
    
    // Simple merge: server state wins if local is empty
    if (!localState.opened_at) {
        localStorage.setItem(sessionKey, JSON.stringify(serverState));
    }
    
    // Restore visual state (e.g. clicked words)
    if (serverState.paragraphs) {
      // Iterate through paragraphs -> sentences -> words to find looked_up_at
      // Since we can't easily match words by ID, we'll rely on surface + occurrence index (or just surface if unique)
      // But wait, the spans have data-word-index.
      // And the 'Big JSON' structure doesn't have word-index.
      // However, the 'Big JSON' was built by iterating words linearly.
      // So we can flatten it back to find which indices were looked up.
      
      let flatIndex = 0;
      const lookedUpIndices = new Set();
      
      for (const para of serverState.paragraphs) {
        if (para.sentences) {
          for (const sent of para.sentences) {
            if (sent.words) {
              for (const w of sent.words) {
                if (w.looked_up_at) {
                  lookedUpIndices.add(flatIndex);
                }
                flatIndex++;
              }
            }
          }
        }
      }
      
      // Apply visual class
      const wordSpans = document.querySelectorAll('.word-span');
      wordSpans.forEach(span => {
        const idx = Number(span.dataset.wordIndex);
        if (lookedUpIndices.has(idx)) {
          span.classList.add('clicked'); // Or whatever style indicates 'clicked'
          span.style.color = '#2563eb'; // Fallback blue if no class defined
        }
      });
    }
    
  } catch (e) {
    console.warn('Failed to restore session:', e);
  }
};

// Call restoration on load
document.addEventListener('DOMContentLoaded', window.arcRestoreSession);

// NOTE: Next button readiness polling is handled per-page (e.g., in home.html template)
// This avoids duplicate polling requests that previously occurred with the global poller
