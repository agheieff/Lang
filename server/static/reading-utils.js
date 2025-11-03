// Shared reading DOM/text helpers
(function(){
  'use strict';

  function normalize(s){ return String(s||'').replace(/[\s\u00A0]+/g,' ').trim(); }

  function serializePlainText(root){
    let out = '';
    function walk(node){
      if(!node) return;
      if(node.nodeType === Node.TEXT_NODE){ out += node.nodeValue || ''; return; }
      if(node.nodeType === Node.ELEMENT_NODE){
        const el = node;
        if(el.hasAttribute && el.hasAttribute('data-word-index')){ out += el.textContent || ''; return; }
        if(el.tagName === 'BR'){ out += '\n'; return; }
        for(let c = el.firstChild; c; c = c.nextSibling){ walk(c); }
      }
    }
    walk(root);
    return out;
  }

  function buildSentenceOffsets(plainText, items){
    let pos = 0; const out = [];
    for(const it of items){
      const src = normalize(it.source);
      if(!src){ out.push({ start: null, end: null, item: it }); continue; }
      let foundAt = plainText.indexOf(it.source, pos);
      if(foundAt < 0){
        const windowEnd = Math.min(plainText.length, pos + 4000);
        const windowStr = normalize(plainText.slice(pos, windowEnd));
        const idxInWin = windowStr.indexOf(src);
        if(idxInWin >= 0){
          let i = pos, seen = 0;
          while(i < windowEnd && seen < idxInWin){
            const ch = plainText[i];
            if(/\s/.test(ch)){ while(i < windowEnd && /\s/.test(plainText[i])) i++; seen++; }
            else { i++; seen++; }
          }
          foundAt = i;
        }
      }
      if(foundAt >= 0){ const endAt = foundAt + (it.source||'').length; out.push({ start: foundAt, end: endAt, item: it }); pos = endAt; }
      else { out.push({ start: null, end: null, item: it }); }
    }
    return out;
  }

  window.arcReadingUtils = Object.freeze({ serializePlainText, buildSentenceOffsets });
})();
