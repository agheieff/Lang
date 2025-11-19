// Reading page: right-click context translation for sentence/paragraph
(function(){
  'use strict';

  function $(id){ return document.getElementById(id); }

  function ready(fn){
    if(document.readyState === 'loading'){
      document.addEventListener('DOMContentLoaded', fn, { once: true });
    } else { fn(); }
  }

  function getTextId(){
    const el = $('reading-text');
    return el && el.dataset && el.dataset.textId ? Number(el.dataset.textId) : 0;
  }

  function ensureTooltip(){
    let el = $('word-tooltip');
    if(!el){
      el = document.createElement('div');
      el.id = 'word-tooltip';
      el.className = 'absolute z-10 bg-white border border-gray-200 rounded-lg shadow p-3 text-sm';
      document.body.appendChild(el);
    }
    return el;
  }

  function serializePlainText(root){
    let out = '';
    function walk(node){
      if(!node) return;
      if(node.nodeType === Node.TEXT_NODE){
        out += node.nodeValue || '';
        return;
      }
      if(node.nodeType === Node.ELEMENT_NODE){
        const el = node;
        if(el.hasAttribute && el.hasAttribute('data-word-index')){
          out += el.textContent || '';
          return; // atomic
        }
        if(el.tagName === 'BR') { out += '\n'; return; }
        for(let c = el.firstChild; c; c = c.nextSibling){ walk(c); }
      }
    }
    walk(root);
    return out;
  }

  function caretRangeFromPoint(x, y){
    const d = document;
    if(d.caretRangeFromPoint){
      try{ return d.caretRangeFromPoint(x, y); }catch(_e){}
    }
    if(d.caretPositionFromPoint){
      try{
        const pos = d.caretPositionFromPoint(x, y);
        if(pos){ const r = d.createRange(); r.setStart(pos.offsetNode, pos.offset); r.collapse(true); return r; }
      }catch(_e){}
    }
    return null;
  }

  function domToCharIndex(root, node, offset){
    // Sum text length until (node, offset); treat word-spans as atomic; BR as \n
    let idx = 0, found = false;
    function measureSubtree(n){
      if(!n) return 0;
      if(n.nodeType === Node.TEXT_NODE){ return (n.nodeValue||'').length; }
      if(n.nodeType === Node.ELEMENT_NODE){
        const el = n;
        if(el.hasAttribute && el.hasAttribute('data-word-index')) return (el.textContent||'').length;
        if(el.tagName === 'BR') return 1;
        let sum = 0; for(let c = el.firstChild; c; c = c.nextSibling){ sum += measureSubtree(c); }
        return sum;
      }
      return 0;
    }
    function walk(n){
      if(found || !n) return;
      if(n.nodeType === Node.TEXT_NODE){
        if(n === node){ idx += Math.min(offset, (n.nodeValue||'').length); found = true; return; }
        idx += (n.nodeValue||'').length; return;
      }
      if(n.nodeType === Node.ELEMENT_NODE){
        const el = n;
        if(el === node){
          // offset is child index within el
          let k = 0, c = el.firstChild; let i = 0;
          while(c && i < offset){ idx += measureSubtree(c); c = c.nextSibling; i++; }
          found = true; return;
        }
        // Treat word spans as atomic unless the caret target is inside them
        if(el.hasAttribute && el.hasAttribute('data-word-index')){
          if(node && (el === node || (el.contains && el.contains(node)))){
            for(let c=el.firstChild; c; c=c.nextSibling){ walk(c); if(found) break; }
            return;
          }
          idx += (el.textContent||'').length; return;
        }
        if(el.tagName === 'BR'){ idx += 1; return; }
        for(let c=el.firstChild; c; c=c.nextSibling){ walk(c); if(found) break; }
      }
    }
    walk(root);
    return idx;
  }

  function findSentenceBounds(text, at){
    const n = text.length;
    if(n === 0) return { start: 0, end: 0 };
    const stops = /[\.!?。！？]/;
    // backward to prev stop or line start
    let s = at;
    while(s > 0){
      const ch = text[s-1];
      if(ch === '\n') break;
      if(stops.test(ch)) break;
      s--;
    }
    // skip following spaces/newlines
    while(s < n && (text[s] === ' ' || text[s] === '\t' || text[s] === '\n')) s++;
    let e = at;
    while(e < n){
      const ch = text[e];
      if(ch === '\n') break;
      if(stops.test(ch)) { e++; break; }
      e++;
    }
    return { start: Math.max(0, s), end: Math.min(n, e) };
  }

  function findParagraphBounds(text, at){
    const n = text.length;
    if(n === 0) return { start: 0, end: 0 };
    // Paragraphs split by blank line (\n\n+)
    let s = at;
    while(s > 0){
      if(text[s-1] === '\n' && text[s] === '\n') break;
      s--;
    }
    // move to first non-newline
    while(s < n && text[s] === '\n') s++;
    let e = at;
    while(e < n-1){
      if(text[e] === '\n' && text[e+1] === '\n') break;
      e++;
    }
    // include until next newline boundary
    while(e < n && text[e] !== '\n') e++;
    return { start: Math.max(0, s), end: Math.min(n, e) };
  }

  function normalize(s){ return String(s||'').replace(/[\s\u00A0]+/g,' ').trim(); }

  async function loadTranslations(textId, unit, wait){
    const w = (typeof wait === 'number' && wait > 0) ? wait : 0;
    const url = `/reading/${textId}/translations?unit=${encodeURIComponent(unit)}&wait=${String(w)}`;
    try{
      const res = await fetch(url, { headers: { 'Accept': 'application/json' } });
      if(!res.ok) return null;
      return await res.json();
    }catch(_e){ return null; }
  }

  // Heuristic mapping removed: we rely on server-provided spans

  function showPopupAt(x, y, html, opts){
    const tip = ensureTooltip();
    tip.innerHTML = html;
    tip.classList.remove('hidden');
    tip.style.position = 'absolute';
    tip.style.maxWidth = (opts && opts.maxWidth) || '680px';
    tip.style.whiteSpace = 'pre-wrap';
    // position near click, constrained to viewport
    const vw = window.innerWidth, vh = window.innerHeight;
    // temporarily make visible to measure
    tip.style.visibility = 'hidden';
    tip.style.left = '0px'; tip.style.top = '0px';
    // force layout
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let nx = Math.min(x + 4, vw - w - 8);
    let ny = Math.min(y + 4, vh - h - 8);
    if(nx < 8) nx = 8; if(ny < 8) ny = 8;
    tip.style.left = (nx + window.scrollX) + 'px';
    tip.style.top = (ny + window.scrollY) + 'px';
    tip.style.visibility = 'visible';
  }

  function escapeHtml(s){
    return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]));
  }

  async function showUnitTranslationAtEvent(ev, unit){
    const textEl = $('reading-text');
    if(!textEl) return;
    const textId = getTextId(); if(!textId) return;
    const range = caretRangeFromPoint(ev.clientX, ev.clientY);
    let charIdx = 0;
    if(range){
      charIdx = domToCharIndex(textEl, range.startContainer, range.startOffset);
    }
    const plain = serializePlainText(textEl);
    const bounds = unit === 'sentence' ? findSentenceBounds(plain, charIdx) : findParagraphBounds(plain, charIdx);
    let src = normalize(plain.slice(bounds.start, bounds.end));
    // Show loading popup immediately (even if src is empty; we may still resolve via char mapping)
    showPopupAt(ev.clientX, ev.clientY, '<div class="text-gray-500">Loading translation…</div>', { maxWidth: unit==='paragraph' ? '740px' : '640px' });

    if(unit === 'sentence'){
      let data = await loadTranslations(textId, 'sentence', 0);
      let items = (data && Array.isArray(data.items)) ? data.items : [];
      if(!items.length){ data = await loadTranslations(textId, 'sentence', 10); items = (data && Array.isArray(data.items)) ? data.items : []; }
      const withSpans = items.filter(it => Number.isInteger(it.start) && Number.isInteger(it.end) && it.end > it.start);
      if(!withSpans.length){
        showPopupAt(ev.clientX, ev.clientY, '<div class="text-gray-500">No translation available yet.</div>', { maxWidth: '640px' });
        return;
      }
      const ci = (bounds.end > bounds.start) ? Math.max(bounds.start, Math.min(bounds.end - 1, charIdx)) : charIdx;
      let match = null; let bestDist = Infinity;
      for(const it of withSpans){
        if(ci >= it.start && ci < it.end){ match = it; break; }
        const mid = (it.start + it.end) / 2; const d = Math.abs(ci - mid);
        if(d < bestDist){ bestDist = d; match = it; }
      }
      if(!match){
        showPopupAt(ev.clientX, ev.clientY, '<div class="text-gray-500">No translation available yet.</div>', { maxWidth: '640px' });
        return;
      }
      const html = `<div class="text-sm">${escapeHtml(match.translation||'')}</div>`;
      showPopupAt(ev.clientX, ev.clientY, html, { maxWidth: '640px' });
      return;
    }

    // Paragraph: try paragraph unit first
    let pdata = await loadTranslations(textId, 'paragraph');
    if(pdata && Array.isArray(pdata.items) && pdata.items.length){
      const items = pdata.items;
      let best = null;
      for(const it of items){ if(normalize(it.source) === src){ best = it; break; } }
      if(!best){ for(const it of items){ const s2 = normalize(it.source); if(s2.includes(src) || src.includes(s2)) { best = it; break; } } }
      if(best){
        const html = `<div class="text-sm">${escapeHtml(best.translation||'')}</div>`;
        showPopupAt(ev.clientX, ev.clientY, html, { maxWidth: '740px' });
        return;
      }
    }
    // Fallback: stitch sentence translations for sentences inside this paragraph span
    const sdata = await loadTranslations(textId, 'sentence');
    const sits = (sdata && Array.isArray(sdata.items)) ? sdata.items : [];
    const parts = [];
    for(const it of sits){
      if(!Number.isInteger(it.start) || !Number.isInteger(it.end)) continue;
      if(it.start >= bounds.end) break;
      if(it.end <= bounds.start) continue;
      if(it.translation) parts.push(String(it.translation));
    }
    if(!parts.length){
      showPopupAt(ev.clientX, ev.clientY, '<div class="text-gray-500">No translation available yet.</div>', { maxWidth: '740px' });
      return;
    }
    showPopupAt(ev.clientX, ev.clientY, `<div class="text-sm">${escapeHtml(parts.join(' '))}</div>`, { maxWidth: '740px' });
  }

  // Register context menu rule for #reading-text
  function registerRule(){
    if(!window.arcContextMenu) return;
    // avoid duplicate registrations on swaps
    if(window.__arcCtxUnregReading){ try{ window.__arcCtxUnregReading(); }catch(_e){} }
    window.__arcCtxUnregReading = window.arcContextMenu.register({
      selector: '#reading-text',
      when: () => true,
      items: (ctx) => [
        { label: 'Sentence translation', onClick: () => showUnitTranslationAtEvent(ctx.event, 'sentence') },
        { label: 'Paragraph translation', onClick: () => showUnitTranslationAtEvent(ctx.event, 'paragraph') },
      ]
    });
  }

  // Title: use structured translation if available, fallback to word translations
  function loadEmbeddedTitleWords(){
    try{
      const el = document.getElementById('reading-title-words-json');
      if(!el) return [];
      const data = JSON.parse(el.textContent || '[]');
      return Array.isArray(data) ? data : [];
    }catch(_e){ return []; }
  }

  function loadStructuredTitleTranslation(){
    try{
      const el = document.getElementById('reading-title-translation');
      if(!el) return null;
      const translation = JSON.parse(el.textContent || 'null');
      return translation ? String(translation).trim() : null;
    }catch(_e){ return null; }
  }

  function buildTitleTranslation(){
    // Prefer structured title translation from LLM response
    const structuredTranslation = loadStructuredTitleTranslation();
    if(structuredTranslation) return structuredTranslation;
    
    // Fallback to word-by-word translation
    const words = loadEmbeddedTitleWords();
    if(!words || !words.length) return null;
    const parts = [];
    for(const w of words){
      if(w && typeof w.translation === 'string' && w.translation.trim()){
        parts.push(String(w.translation).trim());
      }
    }
    if(!parts.length) return null;
    return parts.join(' ');
  }

  function showTitleTranslationAtEvent(ev){
    const tr = buildTitleTranslation();
    if(!tr){
      showPopupAt(ev.clientX, ev.clientY, '<div class="text-gray-500">No translation available yet.</div>', { maxWidth: '520px' });
      return;
    }
    const html = `<div class="text-sm">${escapeHtml(tr)}</div>`;
    showPopupAt(ev.clientX, ev.clientY, html, { maxWidth: '520px' });
  }

  function registerTitleRule(){
    if(!window.arcContextMenu) return;
    const titleEl = document.getElementById('reading-title');
    if(!titleEl) return;
    if(window.__arcCtxUnregTitle){ try{ window.__arcCtxUnregTitle(); }catch(_e){} }
    window.__arcCtxUnregTitle = window.arcContextMenu.register({
      selector: '#reading-title',
      when: () => true,
      items: (ctx) => [
        { label: 'Translate title', onClick: () => showTitleTranslationAtEvent(ctx.event) },
      ]
    });
  }

  function init(){ registerRule(); registerTitleRule(); }

  ready(init);
  document.addEventListener('htmx:afterSwap', function(ev){
    if(ev && ev.detail && ev.detail.elt && ev.detail.elt.id === 'current-reading'){ registerRule(); registerTitleRule(); }
  });
})();
