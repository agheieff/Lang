/**
 * Server-Sent Events handler for reading page.
 * Manages real-time updates for text generation and translation progress.
 */
(function() {
    'use strict';
    
    // Singleton instance
    let instance = null;
    
    class ReadingSSEManager {
        constructor() {
            if (instance) {
                return instance;
            }
            
            this.eventSource = null;
            this.textId = null;
            this.accountId = null;
            this.isReady = false;
            this.retryCount = 0;
            this.maxRetries = 5;
            this.retryDelay = 1000; // 1 second
            
            // Event handlers
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
        
        /**
         * Initialize SSE connection for a text.
         * @param {Object} config - Configuration object
         * @param {number} config.textId - The text ID to monitor
         * @param {number} config.accountId - The user account ID
         * @param {string} config.sseEndpoint - SSE endpoint URL
         * @param {boolean} config.isReady - Whether text is initially ready
         */
        init(config) {
            if (!config || !config.textId || !config.accountId || !config.sseEndpoint) {
                console.error('[SSE] Missing required configuration');
                return false;
            }
            
            // Don't reconnect if already connected to same text
            if (this.eventSource && this.textId === config.textId) {
                return true;
            }
            
            // Close existing connection if different
            if (this.eventSource) {
                this.disconnect();
            }
            
            this.textId = config.textId;
            this.accountId = config.accountId;
            this.isReady = config.isReady || false;
            this.retryCount = 0;
            
            // Build SSE URL
            const url = new URL(config.sseEndpoint, window.location.origin);
            
            // Connect
            this.connect(url);
            return true;
        }
        
        /**
         * Establish SSE connection.
         */
        connect(url) {
            try {
                console.log(`[SSE] Connecting to ${url}`);
                
                this.eventSource = new EventSource(url);
                
                // Connection events
                this.eventSource.onopen = (event) => {
                    console.log('[SSE] Connected');
                    this.retryCount = 0;
                    
                    if (this.handlers.onConnected) {
                        this.handlers.onConnected(event);
                    }
                };
                
                this.eventSource.onerror = (event) => {
                    console.error('[SSE] Connection error:', event);
                    this.eventSource.close();
                    this.eventSource = null;
                    
                    // Try to reconnect with exponential backoff
                    if (this.retryCount < this.maxRetries) {
                        const delay = this.retryDelay * Math.pow(2, this.retryCount);
                        console.log(`[SSE] Reconnecting in ${delay}ms...`);
                        
                        setTimeout(() => {
                            this.retryCount++;
                            this.connect(url);
                        }, delay);
                    } else {
                        console.error('[SSE] Max retries reached, giving up');
                        if (this.handlers.onDisconnected) {
                            this.handlers.onDisconnected(event);
                        }
                    }
                };
                
                // Server-sent event types
                this.eventSource.addEventListener('generation_started', (event) => {
                    console.log('[SSE] Generation started:', JSON.parse(event.data));
                    if (this.handlers.onGenerationStarted) {
                        this.handlers.onGenerationStarted(JSON.parse(event.data));
                    }
                });
                
                this.eventSource.addEventListener('content_ready', (event) => {
                    const payload = JSON.parse(event.data);
                    console.log('[SSE] Content ready:', payload);
                    if (this.handlers.onContentReady) {
                        this.handlers.onContentReady(payload);
                    }
                });
                
                this.eventSource.addEventListener('translations_ready', (event) => {
                    console.log('[SSE] Translations ready:', JSON.parse(event.data));
                    this.isReady = true;
                    if (this.handlers.onTranslationsReady) {
                        this.handlers.onTranslationsReady(JSON.parse(event.data));
                    }
                });
                
                this.eventSource.addEventListener('generation_failed', (event) => {
                    console.log('[SSE] Generation failed:', JSON.parse(event.data));
                    if (this.handlers.onGenerationFailed) {
                        this.handlers.onGenerationFailed(JSON.parse(event.data));
                    }
                });
                
                this.eventSource.addEventListener('next_ready', (event) => {
                    console.log('[SSE] Next text ready:', JSON.parse(event.data));
                    if (this.handlers.onNextReady) {
                        this.handlers.onNextReady(JSON.parse(event.data));
                    }
                });
                
                // Default message handler (for heartbeats, etc.)
                this.eventSource.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'connected') {
                        console.log('[SSE] Connection confirmed');
                    } else if (data.type === 'heartbeat') {
                        // Heartbeat - just keep-alive
                    } else {
                        console.log('[SSE] Unknown message:', data);
                    }
                };
                
            } catch (error) {
                console.error('[SSE] Failed to create EventSource:', error);
                if (this.handlers.onDisconnected) {
                    this.handlers.onDisconnected({ error });
                }
            }
        }
        
        /**
         * Disconnect from SSE.
         */
        disconnect() {
            if (this.eventSource) {
                console.log('[SSE] Disconnecting');
                this.eventSource.close();
                this.eventSource = null;
            }
        }
        
        /**
         * Check if currently connected.
         */
        isConnected() {
            return this.eventSource && this.eventSource.readyState === EventSource.OPEN;
        }
        
        /**
         * Set event handlers.
         * @param {Object} handlers - Event handlers
         */
        setHandlers(handlers) {
            this.handlers = { ...this.handlers, ...handlers };
        }
    }
    
    // Public API
    window.ReadingSSEManager = ReadingSSEManager;
    
    // Initialize on page load if we have the necessary data
    document.addEventListener('DOMContentLoaded', function() {
        // Check if we have reading seeds
        const seedsEl = document.getElementById('reading-seeds');
        if (!seedsEl) return;
        
        try {
            const seeds = JSON.parse(seedsEl.textContent);
            if (!seeds.sse_endpoint || !seeds.text_id || !seeds.account_id) {
                return;
            }
            
            // Create SSE manager
            const sse = new ReadingSSEManager();
            
            // Set up handlers
            sse.setHandlers({
                onGenerationStarted: (data) => {
                    // Show generating state
                    const statusEl = document.getElementById('next-status');
                    if (statusEl) {
                        statusEl.textContent = 'Generating content...';
                        statusEl.className = 'ml-3 text-sm text-blue-500';
                    }
                },
                
                onContentReady: (data) => {
                    // Content is ready, waiting for translations
                    const statusEl = document.getElementById('next-status');
                    if (statusEl) {
                        statusEl.textContent = 'Processing translations...';
                        statusEl.className = 'ml-3 text-sm text-yellow-500';
                    }
                    
                    // Only refresh if this event refers to the current text
                    const textEl = document.getElementById('reading-text');
                    const curId = textEl && textEl.dataset ? Number(textEl.dataset.textId) : null;
                    if (curId && data && Number(data.text_id) === curId) {
                        if (window.ReadingController && window.ReadingController.requestRefresh) {
                            window.ReadingController.requestRefresh('content_ready_sse');
                        } else if (window.htmx && window.htmx.ajax) {
                            window.htmx.ajax('GET', '/reading/current', {
                                target: '#current-reading',
                                swap: 'innerHTML',
                            });
                        } else {
                            window.location.reload();
                        }
                    }
                },
                
                onTranslationsReady: (data) => {
                    // Everything is ready
                    const statusEl = document.getElementById('next-status');
                    if (statusEl) {
                        statusEl.textContent = 'Ready';
                        statusEl.className = 'ml-3 text-sm text-green-500';
                    }
                    
                    // Enable next button
                    const nextBtn = document.getElementById('next-btn');
                    if (nextBtn) {
                        nextBtn.disabled = false;
                        nextBtn.setAttribute('aria-disabled', 'false');
                    }
                    
                    // Load words data if not already present
                    loadWordsIfNeeded();
                    
                    // Disconnect SSE since we're done
                    sse.disconnect();
                },
                
                onGenerationFailed: (data) => {
                    // Show error state
                    const statusEl = document.getElementById('next-status');
                    if (statusEl) {
                        statusEl.textContent = 'Generation failed';
                        statusEl.className = 'ml-3 text-sm text-red-500';
                    }
                    
                    if (data.error) {
                        console.error('[SSE] Generation failed:', data.error);
                    }
                    
                    // Retry button or other error handling could go here
                },
                
                onNextReady: (data) => {
                    // Next text is fully ready - enable the Next button
                    const statusEl = document.getElementById('next-status');
                    if (statusEl) {
                        statusEl.textContent = 'Next text ready';
                        statusEl.className = 'ml-3 text-sm text-green-500';
                    }
                    
                    const nextBtn = document.getElementById('next-btn');
                    if (nextBtn) {
                        nextBtn.disabled = false;
                        nextBtn.setAttribute('aria-disabled', 'false');
                    }
                },
                
                onDisconnected: (event) => {
                    // Handle disconnection
                    const statusEl = document.getElementById('next-status');
                    if (statusEl) {
                        statusEl.textContent = 'Connection lost';
                        statusEl.className = 'ml-3 text-sm text-gray-500';
                    }
                }
            });
            
            // Initialize connection
            sse.init({
                textId: seeds.text_id,
                accountId: seeds.account_id,
                sseEndpoint: seeds.sse_endpoint,
                isReady: seeds.ready
            });
            
            // Store reference globally
            window.readingSSE = sse;
            
        } catch (error) {
            console.error('[SSE] Failed to initialize:', error);
        }
    });
    
    // Helper function to load words if needed
    function loadWordsIfNeeded() {
        const wordsJson = document.getElementById('reading-words-json');
        if (wordsJson && wordsJson.textContent.trim() === '[]') {
            // Words not loaded yet, fetch them
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
    
})();
