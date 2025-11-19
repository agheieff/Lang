(function () {
  'use strict';

  class ReadingController {
    constructor() {
      this.refreshInFlight = false;
      this.pendingReason = null;
      this.homeSSE = null;
      this.htmxHooksInstalled = false;
      this.pollTimerId = null;
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

    initHomeSSE() {
      if (this.homeSSE) return;
      const container = document.getElementById('current-reading');
      if (!container) return;

      try {
        const es = new EventSource('/reading/events/sse');
        this.homeSSE = es;

        es.addEventListener('text_ready', (event) => {
          try {
            // payload not currently used beyond triggering refresh
            void JSON.parse(event.data || 'null');
          } catch (_e) {}
          this.requestRefresh('home_text_ready');
        });

        es.addEventListener('content_ready', (event) => {
          try {
            void JSON.parse(event.data || 'null');
          } catch (_e) {}
          this.requestRefresh('home_content_ready');
        });

        es.addEventListener('translations_ready', (event) => {
          try {
            void JSON.parse(event.data || 'null');
          } catch (_e) {}
          this.requestRefresh('home_translations_ready');
        });

        es.addEventListener('error', (event) => {
          console.error('[HOME SSE] Connection error', event);
        });

        window.addEventListener('beforeunload', () => {
          try { es.close(); } catch (_e) {}
        });

        this._startEmptyPoll();
      } catch (e) {
        console.error('[HOME SSE] Failed to init SSE', e);
      }
    }

    requestRefresh(reason) {
      const container = document.getElementById('current-reading');
      if (!container) return;

      if (this.refreshInFlight) {
        this.pendingReason = reason || this.pendingReason;
        return;
      }
      this.refreshInFlight = true;
      this._doFetchRefresh(container);
    }

    async _doFetchRefresh(container) {
      try {
        const res = await fetch('/reading/current', {
          headers: {
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
          },
        });
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const html = await res.text();
        const doc = new DOMParser().parseFromString(html, 'text/html');
        const block = doc.querySelector('#reading-block');
        if (!block) throw new Error('missing reading block');
        container.innerHTML = block.innerHTML;
        try {
          if (window.arcInitReadingBlock) window.arcInitReadingBlock();
        } catch (_e) {}
        this._onRefreshFinished(null);
      } catch (err) {
        console.error('[ReadingController] Refresh failed', err);
        this._onRefreshFinished(err || new Error('refresh-failed'));
      }
    }

    _onRefreshFinished(err) {
      if (!this.refreshInFlight) return;
      this.refreshInFlight = false;

      if (err) {
        this._showError();
      }

       // Stop polling once we have actual text
       try {
         const container = document.getElementById('current-reading');
         const hasText = container && container.querySelector('#reading-text');
         if (hasText && this.pollTimerId) {
           clearInterval(this.pollTimerId);
           this.pollTimerId = null;
         }
       } catch (_e) {}

      if (this.pendingReason) {
        const r = this.pendingReason;
        this.pendingReason = null;
        this.requestRefresh(r);
      }
    }

    _showError() {
      const container = document.getElementById('current-reading');
      if (!container) return;

      const hasBlock = !!container.querySelector('#reading-block');
      if (hasBlock) {
        let banner = container.querySelector('[data-reading-error-banner]');
        if (!banner) {
          banner = document.createElement('div');
          banner.setAttribute('data-reading-error-banner', '1');
          banner.className = 'mt-4 text-center';
          banner.innerHTML = (
            '<p class="text-red-500 text-sm">Couldn\u2019t refresh the text. ' +
            '<button type="button" class="underline text-blue-600">Try again</button>' +
            '</p>'
          );
          container.appendChild(banner);
          const btn = banner.querySelector('button');
          if (btn) {
            btn.addEventListener('click', () => this.requestRefresh('user_retry'));
          }
        }
        return;
      }

      container.innerHTML = (
        '<div class="text-center py-8" data-reading-error="1">' +
        '<p class="text-red-500">Couldn\u2019t load the text right now.</p>' +
        '<button type="button" class="mt-4 px-4 py-2 bg-blue-600 text-white rounded-lg">Try again</button>' +
        '</div>'
      );
      const btn = container.querySelector('button');
      if (btn) {
        btn.addEventListener('click', () => this.requestRefresh('user_retry'));
      }
    }

    _startEmptyPoll() {
      if (this.pollTimerId) return;
      this.pollTimerId = setInterval(() => {
        try {
          const container = document.getElementById('current-reading');
          if (!container) return;
          const hasText = container.querySelector('#reading-text');
          if (hasText) {
            clearInterval(this.pollTimerId);
            this.pollTimerId = null;
            return;
          }
          this.requestRefresh('poll_empty');
        } catch (_e) {}
      }, 5000);
    }
  }

  window.ReadingController = new ReadingController();
})();
