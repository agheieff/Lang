/**
 * Server-Sent Events handler for reading page.
 * Single source of truth for "Next text" button state.
 * 
 * Readiness flow:
 * 1. On page load, check seeds.is_next_ready - enable if true
 * 2. Connect to SSE and listen for next_ready event
 * 3. If SSE disconnects, fall back to polling /reading/next/ready
 */
(function() {
    'use strict';
    
    let instance = null;
    let pollingInterval = null;
    const POLL_INTERVAL_MS = 5000;
    
    class ReadingSSEManager {
        constructor() {
            if (instance) return instance;
            
            this.eventSource = null;
            this.textId = null;
            this.accountId = null;
            this.isNextReady = false;
            this.retryCount = 0;
            this.maxRetries = 3;
            this.retryDelay = 2000;
            this.sseUrl = null;
            
            this.handlers = {
                onGenerationStarted: null,
                onContentReady: null,
                onTranslationsReady: null,
                onGenerationFailed: null,
                onNextReady: null,
                onConnected: null,
                onDisconnected: null
            };
            
            instance = this;
        }
        
        init(config) {
            if (!config || !config.textId || !config.accountId || !config.sseEndpoint) {
                console.error('[SSE] Missing required configuration');
                return false;
            }
            
            if (this.eventSource && this.textId === config.textId) {
                return true;
            }
            
            if (this.eventSource) {
                this.disconnect();
            }
            
            this.textId = config.textId;
            this.accountId = config.accountId;
            this.isNextReady = config.isNextReady || false;
            this.retryCount = 0;
            this.sseUrl = new URL(config.sseEndpoint, window.location.origin);
            
            this.connect();
            return true;
        }
        
        connect() {
            if (!this.sseUrl) return;
            
            try {
                console.log(`[SSE] Connecting to ${this.sseUrl}`);
                this.eventSource = new EventSource(this.sseUrl);
                
                this.eventSource.onopen = () => {
                    console.log('[SSE] Connected');
                    this.retryCount = 0;
                    stopPolling();
                    if (this.handlers.onConnected) {
                        this.handlers.onConnected();
                    }
                };
                
                this.eventSource.onerror = () => {
                    console.warn('[SSE] Connection error');
                    this.eventSource.close();
                    this.eventSource = null;
                    
                    if (this.retryCount < this.maxRetries) {
                        const delay = this.retryDelay * Math.pow(2, this.retryCount);
                        console.log(`[SSE] Reconnecting in ${delay}ms...`);
                        setTimeout(() => {
                            this.retryCount++;
                            this.connect();
                        }, delay);
                    } else {
                        console.log('[SSE] Max retries reached, falling back to polling');
                        startPolling();
                        if (this.handlers.onDisconnected) {
                            this.handlers.onDisconnected();
                        }
                    }
                };
                
                this.eventSource.addEventListener('generation_started', (event) => {
                    const data = JSON.parse(event.data);
                    console.log('[SSE] Generation started:', data);
                    if (this.handlers.onGenerationStarted) {
                        this.handlers.onGenerationStarted(data);
                    }
                });
                
                this.eventSource.addEventListener('content_ready', (event) => {
                    const data = JSON.parse(event.data);
                    console.log('[SSE] Content ready:', data);
                    if (this.handlers.onContentReady) {
                        this.handlers.onContentReady(data);
                    }
                });
                
                this.eventSource.addEventListener('translations_ready', (event) => {
                    const data = JSON.parse(event.data);
                    console.log('[SSE] Translations ready:', data);
                    if (this.handlers.onTranslationsReady) {
                        this.handlers.onTranslationsReady(data);
                    }
                });
                
                this.eventSource.addEventListener('generation_failed', (event) => {
                    const data = JSON.parse(event.data);
                    console.log('[SSE] Generation failed:', data);
                    if (this.handlers.onGenerationFailed) {
                        this.handlers.onGenerationFailed(data);
                    }
                });
                
                this.eventSource.addEventListener('next_ready', (event) => {
                    const data = JSON.parse(event.data);
                    console.log('[SSE] Next text ready:', data);
                    this.isNextReady = true;
                    enableNextButton(data.next_ready_reason || 'both');
                    if (this.handlers.onNextReady) {
                        this.handlers.onNextReady(data);
                    }
                });
                
                this.eventSource.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === 'heartbeat') {
                            // Keep-alive, ignore
                        }
                    } catch (e) {
                        // Ignore parse errors for heartbeats
                    }
                };
                
            } catch (error) {
                console.error('[SSE] Failed to create EventSource:', error);
                startPolling();
            }
        }
        
        disconnect() {
            if (this.eventSource) {
                console.log('[SSE] Disconnecting');
                this.eventSource.close();
                this.eventSource = null;
            }
            stopPolling();
        }
        
        isConnected() {
            return this.eventSource && this.eventSource.readyState === EventSource.OPEN;
        }
        
        setHandlers(handlers) {
            this.handlers = { ...this.handlers, ...handlers };
        }
    }
    
    // Button control functions
    function enableNextButton(reason) {
        const nextBtn = document.getElementById('next-btn');
        const statusEl = document.getElementById('next-status');
        
        if (nextBtn) {
            nextBtn.disabled = false;
            nextBtn.setAttribute('aria-disabled', 'false');
        }
        
        if (statusEl) {
            const statusText = {
                'both': 'Ready',
                'grace': 'Ready',
                'content_only': 'Ready (text only)',
            }[reason] || 'Ready';
            
            statusEl.textContent = statusText;
            statusEl.className = 'ml-3 text-sm text-green-500';
        }
    }
    
    function updateStatus(text, colorClass) {
        const statusEl = document.getElementById('next-status');
        if (statusEl) {
            statusEl.textContent = text;
            statusEl.className = `ml-3 text-sm ${colorClass}`;
        }
    }
    
    // Polling fallback
    function startPolling() {
        if (pollingInterval) return;
        
        console.log('[POLL] Starting polling fallback');
        pollingInterval = setInterval(async () => {
            try {
                const response = await fetch('/reading/next/ready');
                if (!response.ok) return;
                
                const data = await response.json();
                if (data.ready) {
                    console.log('[POLL] Next text ready:', data);
                    enableNextButton(data.ready_reason || 'both');
                    stopPolling();
                }
            } catch (error) {
                console.warn('[POLL] Poll failed:', error);
            }
        }, POLL_INTERVAL_MS);
    }
    
    function stopPolling() {
        if (pollingInterval) {
            console.log('[POLL] Stopping polling');
            clearInterval(pollingInterval);
            pollingInterval = null;
        }
    }
    
    // Helper function to load words if needed
    function loadWordsIfNeeded() {
        const wordsJson = document.getElementById('reading-words-json');
        if (wordsJson && wordsJson.textContent.trim() === '[]') {
            const textEl = document.getElementById('reading-text');
            const textId = textEl ? textEl.dataset.textId : null;
            
            if (textId) {
                fetch(`/reading/${textId}/words`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.words && data.words.length > 0) {
                            if (window.arcApplyWords) window.arcApplyWords(data.words);
                        }
                    })
                    .catch(error => {
                        console.error('Failed to load words:', error);
                    });
            }
        }
    }
    
    // Public API
    window.ReadingSSEManager = ReadingSSEManager;
    
    // Initialize on page load
    document.addEventListener('DOMContentLoaded', function() {
        const seedsEl = document.getElementById('reading-seeds');
        if (!seedsEl) return;
        
        try {
            const seeds = JSON.parse(seedsEl.textContent);
            if (!seeds.sse_endpoint || !seeds.text_id || !seeds.account_id) {
                return;
            }
            
            // Check if next text is already ready from server render
            if (seeds.is_next_ready) {
                console.log('[SSE] Next text already ready on load');
                enableNextButton(seeds.next_ready_reason || 'both');
            }
            
            // Create SSE manager
            const sse = new ReadingSSEManager();
            
            // Set up handlers
            sse.setHandlers({
                onGenerationStarted: () => {
                    updateStatus('Generating...', 'text-blue-500');
                },
                
                onContentReady: (data) => {
                    updateStatus('Processing translations...', 'text-yellow-500');
                    
                    // Refresh if this is for current text
                    const textEl = document.getElementById('reading-text');
                    const curId = textEl && textEl.dataset ? Number(textEl.dataset.textId) : null;
                    if (curId && data && Number(data.text_id) === curId) {
                        if (window.ReadingController && window.ReadingController.requestRefresh) {
                            window.ReadingController.requestRefresh('content_ready_sse');
                        }
                    }
                },
                
                onTranslationsReady: () => {
                    loadWordsIfNeeded();
                },
                
                onGenerationFailed: (data) => {
                    updateStatus('Generation failed', 'text-red-500');
                    console.error('[SSE] Generation failed:', data.error);
                },
                
                onDisconnected: () => {
                    // Polling started automatically in connect() error handler
                    updateStatus('Checking...', 'text-gray-500');
                }
            });
            
            // Initialize connection
            sse.init({
                textId: seeds.text_id,
                accountId: seeds.account_id,
                sseEndpoint: seeds.sse_endpoint,
                isNextReady: seeds.is_next_ready
            });
            
            // Store reference globally
            window.readingSSE = sse;
            
        } catch (error) {
            console.error('[SSE] Failed to initialize:', error);
            // Fall back to polling if SSE init fails
            startPolling();
        }
    });
    
    // Clean up on page unload
    window.addEventListener('beforeunload', function() {
        if (window.readingSSE) {
            window.readingSSE.disconnect();
        }
        stopPolling();
    });
    
    // Also initialize after HTMX swaps (for home page where reading-seeds is loaded via HTMX)
    function tryInitSSE() {
        const seedsEl = document.getElementById('reading-seeds');
        if (!seedsEl) return;
        
        // Don't reinitialize if already connected
        if (window.readingSSE && window.readingSSE.isConnected()) return;
        
        try {
            const seeds = JSON.parse(seedsEl.textContent);
            if (!seeds.sse_endpoint || !seeds.text_id || !seeds.account_id) return;
            
            // Check if next text is already ready from server render
            if (seeds.is_next_ready) {
                console.log('[SSE] Next text already ready on load');
                enableNextButton(seeds.next_ready_reason || 'both');
            }
            
            // Create SSE manager if not exists
            const sse = window.readingSSE || new ReadingSSEManager();
            
            // Set up handlers
            sse.setHandlers({
                onGenerationStarted: () => {
                    updateStatus('Generating...', 'text-blue-500');
                },
                
                onContentReady: (data) => {
                    updateStatus('Processing translations...', 'text-yellow-500');
                    
                    // Refresh if this is for current text
                    const textEl = document.getElementById('reading-text');
                    const curId = textEl && textEl.dataset ? Number(textEl.dataset.textId) : null;
                    if (curId && data && Number(data.text_id) === curId) {
                        if (window.ReadingController && window.ReadingController.requestRefresh) {
                            window.ReadingController.requestRefresh('content_ready_sse');
                        }
                    }
                },
                
                onTranslationsReady: () => {
                    loadWordsIfNeeded();
                },
                
                onGenerationFailed: (data) => {
                    updateStatus('Generation failed', 'text-red-500');
                    console.error('[SSE] Generation failed:', data.error);
                },
                
                onDisconnected: () => {
                    updateStatus('Checking...', 'text-gray-500');
                }
            });
            
            // Initialize connection
            sse.init({
                textId: seeds.text_id,
                accountId: seeds.account_id,
                sseEndpoint: seeds.sse_endpoint,
                isNextReady: seeds.is_next_ready
            });
            
            // Store reference globally
            window.readingSSE = sse;
            
        } catch (error) {
            console.error('[SSE] Failed to initialize:', error);
            startPolling();
        }
    }
    
    document.addEventListener('htmx:afterSwap', function(evt) {
        if (evt.detail && evt.detail.target && evt.detail.target.id === 'current-reading') {
            setTimeout(tryInitSSE, 100);
        }
    });
    
    document.addEventListener('htmx:afterSettle', function(evt) {
        if (evt.detail && evt.detail.target && evt.detail.target.id === 'current-reading') {
            setTimeout(tryInitSSE, 100);
        }
    });
    
})();
