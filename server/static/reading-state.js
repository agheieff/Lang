/**
 * Simplified Text State Manager
 * Text state lives in browser memory, client adds profile info, sent back on "Next Text"
 */

(function() {
    'use strict';

    // Single unified state object - starts as global state from server,
    // client adds profile info immediately on load
    let State = null;

    /**
     * Initialize state when page loads
     */
    async function initializeState() {
        const textId = document.getElementById('reading-text')?.getAttribute('data-text-id');
        const accountId = document.getElementById('reading-text')?.getAttribute('data-account-id');
        const profileId = document.getElementById('reading-text')?.getAttribute('data-profile-id');

        if (!textId) {
            console.warn('[TextState] No text_id found');
            return;
        }

        // Fetch global text state from server
        try {
            const response = await fetch(`/reading/${textId}/state`);
            const data = await response.json();

            if (data.status === 'ok') {
                // Add client-side metadata immediately
                State = data.state;
                State.account_id = accountId ? parseInt(accountId) : null;
                State.profile_id = profileId ? parseInt(profileId) : null;
                State.loaded_at = new Date().toISOString();

                console.log('[TextState] Loaded and enriched:', State);
            } else if (data.status === 'not_ready') {
                console.warn('[TextState] Text state not ready yet:', data.message);
                // Retry after a delay
                setTimeout(initializeState, 2000);
            }
        } catch (error) {
            console.error('[TextState] Failed to load:', error);
        }
    }

    /**
     * Save state to server (called on "Next Text")
     */
    async function saveState() {
        if (!State) {
            console.warn('[TextState] No state to save');
            return null;
        }

        // Add saved_at timestamp
        State.saved_at = new Date().toISOString();

        try {
            const response = await fetch('/reading/log-text-state', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(State)
            });

            const result = await response.json();
            console.log('[TextState] Saved:', result);
            return result;
        } catch (error) {
            console.error('[TextState] Save failed:', error);
            return null;
        }
    }

    /**
     * Clear state (after save)
     */
    function clearState() {
        State = null;
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeState);
    } else {
        initializeState();
    }

    // Export to global scope
    window.ArcadiaTextState = {
        saveState,
        clearState,
        getState: () => State
    };

    console.log('[TextState] Manager initialized');
})();
