/**
 * Consolidated Arcadia Reading Application
 * Merges reading functionality: translations, context menu, controller, SSE, and utilities
 * Single cohesive JavaScript application for reading practice.
 */

(function() {
    'use strict';

    // =============================================================================
    // Global State and Configuration
    // =============================================================================
    
    const AppState = {
        currentTextId: null,
        accountId: null,
        isNextReady: false,
        sessionData: {},
        translationCache: {},
        eventSource: null,
        pollingInterval: null,
        refreshInFlight: false,
        pendingReason: null,
        tooltipVisible: false
    };

    const Config = {
        POLL_INTERVAL_MS: 5000,
        SSE_RETRY_DELAY: 2000,
        SSE_MAX_RETRIES: 3,
        TOOLTIP_HIDE_DELAY: 300
    };

    // =============================================================================
    // Utility Functions
    // =============================================================================
    
    function $(id) { 
        return document.getElementById(id); 
    }
    
    function normalize(s) { 
        return String(s || '').replace(/[\\s\\u00A0]+/g, ' ').trim(); 
    }
    
    function serializePlainText(root) {
        let out = '';
        function walk(node) {
            if (!node) return;
            if (node.nodeType === Node.TEXT_NODE) {
                out += node.nodeValue || '';
                return;
            }
            if (node.nodeType === Node.ELEMENT_NODE) {
                const el = node;
                if (el.hasAttribute && el.hasAttribute('data-word-index')) {
                    out += el.textContent || '';
                    return; // atomic
                }
                if (el.tagName === 'BR') { 
                    out += '\\n'; 
                    return; 
                }
                for (let c = el.firstChild; c; c = c.nextSibling) { 
                    walk(c); 
                }
            }
        }
        walk(root);
        return out;
    }
    
    function buildSentenceOffsets(plainText, items) {
        let pos = 0;
        const out = [];
        for (const it of items) {
            const src = normalize(it.source);
            if (!src) { 
                out.push({ start: null, end: null, item: it }); 
                continue; 
            }
            let foundAt = plainText.indexOf(it.source, pos);
            if (foundAt < 0) {
                const windowEnd = Math.min(plainText.length, pos + 4000);
                const windowStr = normalize(plainText.slice(pos, windowEnd));
                const idxInWin = windowStr.indexOf(src);
                if (idxInWin >= 0) {
                    let i = pos, seen = 0;
                    while (i < windowEnd && seen < idxInWin) {
                        const ch = plainText[i];
                        if (/\\s/.test(ch)) { 
                            while (i < windowEnd && /\\s/.test(plainText[i])) i++; 
                            seen++; 
                        } else { 
                            i++; 
                            seen++; 
                        }
                    }
                    foundAt = i;
                }
            }
            if (foundAt >= 0) { 
                const endAt = foundAt + (it.source || '').length; 
                out.push({ start: foundAt, end: endAt, item: it }); 
                pos = endAt; 
            } else { 
                out.push({ start: null, end: null, item: it }); 
            }
        }
        return out;
    }
    
    async function fetchWithTimeout(url, options = {}, timeout = 30000) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);
        
        try {
            const response = await fetch(url, {
                ...options,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            return response;
        } catch (error) {
            clearTimeout(timeoutId);
            throw error;
        }
    }
    
    // =============================================================================
    // Translation Panel Management
    // =============================================================================
    
    function ensureTooltip() {
        let el = $('word-tooltip');
        if (!el) {
            el = document.createElement('div');
            el.id = 'word-tooltip';
            el.className = 'absolute z-10 bg-white border border-gray-200 rounded-lg shadow p-3 text-sm max-w-xs';
            document.body.appendChild(el);
        }
        return el;
    }
    
    function hideTooltip() {
        const tooltip = ensureTooltip();
        tooltip.style.display = 'none';
        AppState.tooltipVisible = false;
    }
    
    function showTooltip(content, x, y) {
        const tooltip = ensureTooltip();
        tooltip.innerHTML = content;
        tooltip.style.display = 'block';
        
        // Position tooltip
        const tooltipRect = tooltip.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        
        let left = x + 10;
        let top = y - tooltipRect.height / 2;
        
        // Adjust if tooltip would go outside viewport
        if (left + tooltipRect.width > viewportWidth) {
            left = x - tooltipRect.width - 10;
        }
        if (top < 10) {
            top = 10;
        }
        if (top + tooltipRect.height > viewportHeight - 10) {
            top = viewportHeight - tooltipRect.height - 10;
        }
        
        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
        AppState.tooltipVisible = true;
    }
    
    async function toggleTranslation() {
        const panel = $('translation-panel');
        if (!panel) return;
        const show = panel.classList.contains('hidden');
        panel.hidden = !show;
        panel.classList.toggle('hidden', !show);

        // Update button state
        const toggleBtn = $('translation-toggle');
        if (toggleBtn) {
            toggleBtn.setAttribute('aria-expanded', String(show));
        }

        // When opening, fetch translation if missing
        if (show) {
            // Track full translation view
            trackTranslationView('full');

            const textEl = $('translation-text');
            if (textEl && !textEl.textContent.trim()) {
                textEl.textContent = 'Loading translation...';

                // Fetch full text translation
                const mainTextEl = $('reading-text');
                const textId = mainTextEl ? mainTextEl.dataset.textId : null;

                if (textId) {
                    try {
                        const res = await fetchWithTimeout(`/reading/${textId}/translations?unit=text`, {
                            headers: { 'Accept': 'application/json' }
                        });

                        if (res.ok) {
                            const data = await res.json();
                            if (data.items && data.items.length > 0) {
                                // Find longest translation (likely body, not title)
                                const items = data.items.sort((a, b) =>
                                    (b.translation?.length || 0) - (a.translation?.length || 0)
                                );
                                const best = items[0];

                                if (best && best.translation) {
                                    textEl.textContent = best.translation;

                                    // Cache it in session data for persistence
                                    const sessionKey = `arc_current_session_${textId}`;
                                    const sessionData = JSON.parse(localStorage.getItem(sessionKey) || '{}');
                                    sessionData.full_translation = best.translation;
                                    localStorage.setItem(sessionKey, JSON.stringify(sessionData));
                                    return;
                                }
                            }
                        }
                        textEl.textContent = 'Translation is not ready yet.';
                    } catch (error) {
                        console.error('Error fetching translation:', error);
                        textEl.textContent = 'Error loading translation.';
                    }
                } else {
                    textEl.textContent = 'No text available.';
                }
            }
        }
    }

    function showContextMenu(event, wordData) {
        event.preventDefault();
        
        // Hide existing menu
        hideContextMenu();
        
        // Create menu
        const menu = document.createElement('div');
        menu.id = 'context-menu';
        menu.className = 'fixed z-50 bg-white border border-gray-200 rounded-lg shadow-lg py-1 min-w-[200px]';
        menu.style.left = event.pageX + 'px';
        menu.style.top = event.pageY + 'px';
        
        // Add menu items
        const items = [
            {
                text: 'View Translation',
                action: () => {
                    if (wordData.translation) {
                        showTooltip(
                            `<div class="font-medium">${wordData.surface}</div><div class="text-gray-600">${wordData.translation}</div>`,
                            event.pageX,
                            event.pageY
                        );
                    }
                }
            },
            {
                text: 'Add to Vocabulary',
                action: () => {
                    // Track word click
                    trackWordClick(wordData);
                }
            }
        ];
        
        items.forEach(item => {
            const menuItem = document.createElement('button');
            menuItem.className = 'w-full text-left px-4 py-2 text-sm hover:bg-gray-100 focus:outline-none focus:bg-gray-100';
            menuItem.textContent = item.text;
            menuItem.addEventListener('click', () => {
                item.action();
                hideContextMenu();
            });
            menu.appendChild(menuItem);
        });
        
        document.body.appendChild(menu);
        
        // Global click handler to hide menu
        setTimeout(() => {
            document.addEventListener('click', hideContextMenu, { once: true });
        }, 100);
    }
    
    function hideContextMenu() {
        const menu = $('context-menu');
        if (menu) {
            menu.remove();
        }
    }
    
    // =============================================================================
    // Server-Sent Events Management
    // =============================================================================
    
    class SSEManager {
        constructor() {
            this.eventSource = null;
            this.retryCount = 0;
            this.isConnecting = false;
        }
        
        connect(textId, accountId) {
            if (this.eventSource) {
                this.eventSource.close();
            }
            
            AppState.textId = textId;
            AppState.accountId = accountId;
            this.isConnecting = true;
            
            const sseUrl = `/reading/sse?text_id=${textId}&account_id=${accountId}`;
            
            try {
                this.eventSource = new EventSource(sseUrl);
                
                this.eventSource.addEventListener('open', () => {
                    console.log('[SSE] Connected');
                    this.retryCount = 0;
                    this.isConnecting = false;
                    this.startPollingFallback();
                });
                
                this.eventSource.addEventListener('error', (event) => {
                    console.error('[SSE] Error:', event);
                    this.handleConnectionError();
                });
                
                this.eventSource.addEventListener('generation_started', (event) => {
                    const data = JSON.parse(event.data);
                    this.onGenerationStarted(data);
                });
                
                this.eventSource.addEventListener('content_ready', (event) => {
                    const data = JSON.parse(event.data);
                    this.onContentReady(data);
                });
                
                this.eventSource.addEventListener('translations_ready', (event) => {
                    const data = JSON.parse(event.data);
                    this.onTranslationsReady(data);
                });
                
                this.eventSource.addEventListener('next_ready', (event) => {
                    const data = JSON.parse(event.data);
                    this.onNextReady(data);
                });
                
            } catch (error) {
                console.error('[SSE] Connection error:', error);
                this.handleConnectionError();
            }
        }
        
        handleConnectionError() {
            if (this.retryCount < Config.SSE_MAX_RETRIES) {
                this.retryCount++;
                console.log(`[SSE] Retry ${this.retryCount}/${Config.SSE_MAX_RETRIES}`);
                setTimeout(() => {
                    this.connect(AppState.textId, AppState.accountId);
                }, Config.SSE_RETRY_DELAY);
            } else {
                console.warn('[SSE] Max retries reached, falling back to polling');
                this.fallbackToPolling();
            }
        }
        
        fallbackToPolling() {
            this.startPollingFallback();
        }
        
        startPollingFallback() {
            if (AppState.pollingInterval) {
                clearInterval(AppState.pollingInterval);
            }
            
            AppState.pollingInterval = setInterval(async () => {
                try {
                    const response = await fetchWithTimeout(`/reading/${AppState.textId}/status`);
                    const status = await response.json();
                    
                    if (status?.next_ready !== AppState.isNextReady) {
                        AppState.isNextReady = status.next_ready;
                        this.onNextReady({ next_ready: status.next_ready });
                    }
                } catch (error) {
                    console.error('[Polling] Error checking status:', error);
                }
            }, Config.POLL_INTERVAL_MS);
        }
        
        disconnect() {
            if (this.eventSource) {
                this.eventSource.close();
                this.eventSource = null;
            }
            
            if (AppState.pollingInterval) {
                clearInterval(AppState.pollingInterval);
                AppState.pollingInterval = null;
            }
            
            this.isConnecting = false;
        }
        
        onGenerationStarted(data) {
            console.log('[SSE] Generation started:', data);
        }
        
        onContentReady(data) {
            console.log('[SSE] Content ready:', data);
            // Optionally refresh content
        }
        
        onTranslationsReady(data) {
            console.log('[SSE] Translations ready:', data);
            // Clear translation cache for this text
            if (AppState.textId) {
                delete AppState.translationCache[AppState.textId];
            }
        }
        
        onNextReady(data) {
            console.log('[SSE] Next ready:', data);
            AppState.isNextReady = data.next_ready || false;
            this.updateNextButton();
        }
        
        updateNextButton() {
            const nextBtn = $('next-text-btn');
            if (nextBtn) {
                nextBtn.disabled = !AppState.isNextReady;
                if (AppState.isNextReady) {
                    nextBtn.classList.remove('opacity-50', 'cursor-not-allowed');
                    nextBtn.classList.add('hover:bg-blue-600');
                } else {
                    nextBtn.classList.add('opacity-50', 'cursor-not-allowed');
                    nextBtn.classList.remove('hover:bg-blue-600');
                }
            }
        }
    }
    
    // =============================================================================
    // Page Controller
    // =============================================================================
    
    class ReadingController {
        constructor() {
            this.sseManager = new SSEManager();
            this.htmxHooksInstalled = false;
            this._installHtmxHooks();
        }
        
        _installHtmxHooks() {
            if (!window.htmx || this.htmxHooksInstalled) return;
            this.htmxHooksInstalled = true;

            document.addEventListener('htmx:afterSwap', (ev) => {
                if (!ev || !ev.detail || !ev.detail.elt) return;
                if (ev.detail.elt.id !== 'current-reading') return;
                this._onRefreshFinished(null);
            });

            document.addEventListener('htmx:responseError', (ev) => {
                if (!ev || !ev.detail || !ev.detail.elt) return;
                if (ev.detail.elt.id !== 'current-reading') return;
                this._onRefreshFinished(new Error('htmx-response-error'));
            });
        }
        
        requestRefresh(reason) {
            const container = $('current-reading');
            if (!container) return;

            if (AppState.refreshInFlight) {
                AppState.pendingReason = reason || AppState.pendingReason;
                return;
            }
            
            AppState.refreshInFlight = true;
            this._doFetchRefresh(container);
        }
        
        async _doFetchRefresh(container) {
            try {
                const res = await fetchWithTimeout('/reading/current', {
                    headers: {
                        'Cache-Control': 'no-cache',
                    },
                });
                
                if (!res.ok) {
                    throw new Error(`HTTP ${res.status}`);
                }
                
                const html = await res.text();
                container.innerHTML = html;
                this._onRefreshFinished(null);
                
            } catch (error) {
                console.error('[ReadingController] Refresh failed:', error);
                this._onRefreshFinished(error);
            }
        }
        
        _onRefreshFinished(error) {
            AppState.refreshInFlight = false;
            
            if (error) {
                console.error('[ReadingController] Refresh finished with error:', error);
                return;
            }
            
            // Re-initialize page components
            this._reinitializeComponents();
            
            // If there was a pending refresh request, do it now
            if (AppState.pendingReason) {
                const reason = AppState.pendingReason;
                AppState.pendingReason = null;
                setTimeout(() => this.requestRefresh(reason), 100);
            }
        }
        
        _reinitializeComponents() {
            // Update text ID
            const textEl = $('reading-text');
            if (textEl && textEl.dataset.textId) {
                AppState.textId = parseInt(textEl.dataset.textId);
            }
            
            // Reconnect SSE with new text ID
            const accountEl = document.querySelector('[data-account-id]');
            if (accountEl && accountEl.dataset.accountId && AppState.textId) {
                this.sseManager.connect(AppState.textId, parseInt(accountEl.dataset.accountId));
            }
            
            // Re-attach event listeners
            this._attachEventListeners();
        }
        
        initializePage() {
            // Get initial data
            const textEl = $('reading-text');
            const accountEl = document.querySelector('[data-account-id]');

            if (textEl && textEl.dataset.textId) {
                AppState.textId = parseInt(textEl.dataset.textId);
            }

            if (accountEl && accountEl.dataset.accountId) {
                AppState.accountId = parseInt(accountEl.dataset.accountId);
            }

            // Check initial next button state
            const initialNextReady = document.querySelector('[data-next-ready]');
            if (initialNextReady) {
                AppState.isNextReady = initialNextReady.dataset.nextReady === 'true';
                this.sseManager.updateNextButton();
            }

            // Connect SSE
            if (AppState.textId && AppState.accountId) {
                this.sseManager.connect(AppState.textId, AppState.accountId);
            }

            // Attach event listeners
            this._attachEventListeners();

            // Listen for custom event when words are rendered by inline script
            window.addEventListener('arcadia:words-rendered', () => {
                this._attachEventListeners();
            });
        }
        
        _attachEventListeners() {
            // Translation toggle
            const toggleBtn = $('translation-toggle');
            if (toggleBtn) {
                toggleBtn.addEventListener('click', toggleTranslation);
            }

            // Next text button
            const nextBtn = $('next-text-btn');
            if (nextBtn) {
                nextBtn.addEventListener('click', () => {
                    if (AppState.isNextReady) {
                        window.location.href = '/reading/next';
                    }
                });
            }

            // Note: Word hover/click handlers are now in reading.html inline JS
            // to avoid duplicate event listeners and conflicts

            // Hide tooltip on click outside (works with reading.html tooltips)
            document.addEventListener('click', (e) => {
                const tooltip = document.getElementById('word-tooltip');
                if (tooltip && tooltip.style.display === 'block' && !e.target.closest('[data-word-index]') && !e.target.closest('#word-tooltip')) {
                    tooltip.style.display = 'none';
                }
            });
        }
    }
    
    // =============================================================================
    // Data Tracking and Persistence
    // =============================================================================
    
    function initializeSessionData() {
        if (!AppState.textId) return null;
        
        const sessionKey = `arc_current_session_${AppState.textId}`;
        let sessionData = localStorage.getItem(sessionKey);
        
        if (!sessionData) {
            sessionData = {
                text_id: AppState.textId,
                exposed_at: Date.now(),
                words: [],
                sentences: [],
                full_translation_views: []
            };
            localStorage.setItem(sessionKey, JSON.stringify(sessionData));
            console.log('[Session] Initialized new session with exposed_at:', sessionData.exposed_at);
        } else {
            sessionData = JSON.parse(sessionData);
            console.log('[Session] Loaded existing session:', sessionData);
        }
        
        return sessionKey;
    }
    
    function trackWordClick(wordData) {
        if (!AppState.textId || !AppState.accountId) return;
        
        try {
            const sessionKey = `arc_current_session_${AppState.textId}`;
            initializeSessionData();
            
            const sessionData = JSON.parse(localStorage.getItem(sessionKey) || '{}');
            
            // Initialize arrays if not present
            if (!sessionData.words) {
                sessionData.words = [];
            }
            if (!sessionData.sentences) {
                sessionData.sentences = [];
            }
            if (!sessionData.full_translation_views) {
                sessionData.full_translation_views = [];
            }
            
            // Add or update word interaction
            const existingWord = sessionData.words.find(
                w => w.surface === wordData.surface && w.span_start === wordData.span_start
            );
            
            if (existingWord) {
                existingWord.clicked = true;
                existingWord.click_count = (existingWord.click_count || 0) + 1;
            } else {
                sessionData.words.push({
                    surface: wordData.surface,
                    lemma: wordData.lemma,
                    pos: wordData.pos,
                    span_start: wordData.span_start,
                    span_end: wordData.span_end,
                    clicked: true,
                    click_count: 1,
                    timestamp: Date.now()
                });
            }
            
            // Save updated session data
            localStorage.setItem(sessionKey, JSON.stringify(sessionData));
            
            // Send to server (in background)
            navigator.sendBeacon('/reading/word-click', JSON.stringify({
                text_id: AppState.textId,
                word_data: wordData,
                session_data: sessionData
            }));
            
        } catch (error) {
            console.error('Error tracking word click:', error);
        }
    }
    
    function trackTranslationView(type = 'full', wordData = null) {
        if (!AppState.textId) return;

        try {
            const sessionKey = `arc_current_session_${AppState.textId}`;
            initializeSessionData();
            const sessionData = JSON.parse(localStorage.getItem(sessionKey) || '{}');

            if (type === 'full') {
                // Track full text translation view
                if (!sessionData.full_translation_views) {
                    sessionData.full_translation_views = [];
                }
                sessionData.full_translation_views.push({
                    timestamp: Date.now()
                });
            } else if (type === 'word' && wordData) {
                // Track word translation view
                if (!sessionData.words) {
                    sessionData.words = [];
                }

                // Find and update word with translation_viewed
                const word = sessionData.words.find(
                    w => w.surface === wordData.surface && w.span_start === wordData.span_start
                );

                if (word) {
                    word.translation_viewed = true;
                    word.translation_viewed_at = Date.now();
                }
            }

            localStorage.setItem(sessionKey, JSON.stringify(sessionData));
            console.log(`[Session] Tracked ${type} translation view`);
        } catch (error) {
            console.error('Error tracking translation view:', error);
        }
    }

    async function saveSessionToServer() {
        if (!AppState.textId || !AppState.accountId) return;
        
        try {
            const sessionKey = localStorage.getItem(`arc_current_session_${AppState.textId}`);
            if (!sessionKey) return;
            
            const sessionData = JSON.parse(localStorage.getItem(sessionKey) || '{}');
            
            const response = await fetchWithTimeout('/reading/save-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    text_id: AppState.textId,
                    session_data: sessionData
                })
            });
            
            if (response.ok) {
                // Clear local session data after successful save
                localStorage.removeItem(sessionKey);
                console.log('[Session] Saved to server successfully');
            }
            
        } catch (error) {
            console.error('[Session] Error saving to server:', error);
        }
    }
    
    // =============================================================================
    // Initialization
    // =============================================================================
    
    function initializeApp() {
        console.log('[Arcadia] Initializing reading application');
        
        const controller = new ReadingController();
        
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                controller.initializePage();
            });
        } else {
            controller.initializePage();
        }
        
        // Save session data before page unload
        window.addEventListener('beforeunload', () => {
            saveSessionToServer();
        });
        
        // Save session data periodically
        setInterval(() => {
            saveSessionToServer();
        }, 30000); // Every 30 seconds
        
        // Expose global functions for backward compatibility
        window.arcToggleTranslation = toggleTranslation;
        window.arcReadingUtils = Object.freeze({ 
            serializePlainText, 
            buildSentenceOffsets 
        });
    }
    
    // Start the application
    initializeApp();
    
    // Export for testing or external use
    window.ArcadiaReadingApp = {
        AppState,
        Config,
        SSEManager,
        ReadingController,
        initializeApp
    };
    
})();
