type Token = { t: string; isWord: boolean };

const inputEl = document.getElementById('input') as HTMLTextAreaElement;
const outEl = document.getElementById('output') as HTMLDivElement;
const renderBtn = document.getElementById('render') as HTMLButtonElement;
const srcSel = document.getElementById('src') as HTMLSelectElement;
const tgtSel = document.getElementById('tgt') as HTMLSelectElement;
const popover = document.getElementById('popover') as HTMLDivElement;
const hoverTip = document.getElementById('hover-tip') as HTMLDivElement;

function tokenize(text: string): Token[] {
  const re = /([\p{L}\p{M}]+)|([^\p{L}\p{M}]+)/gu;
  const out: Token[] = [];
  for (const m of text.matchAll(re) as any) out.push({ t: m[0], isWord: !!m[1] });
  return out;
}

async function lookup(source_lang: string, target_lang: string, surface: string) {
  const res = await fetch('/api/lookup', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_lang, target_lang, surface })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function parse(lang: string, text: string) {
  const res = await fetch('/api/parse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lang, text })
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function clearPopover() { popover.style.display = 'none'; }

function showPopover(x: number, y: number, payload: any) {
  const { mode, lemma, translations, label, pos, pronunciation } = payload;
  popover.style.left = `${x}px`;
  popover.style.top = `${y}px`;
  let html = '';
  const header = `<div><strong>${lemma ?? ''}</strong>${pos ? ` <span style="color:#888;font-weight:normal">(${pos})</span>` : ''}</div>`;
  if (mode === 'translation' && translations?.length) {
    const items = translations.slice(0, 6);
    html += header;
    // Chinese: show full-token pinyin (diacritics) when available
    if (pronunciation?.pinyin) html += `<div class="label">${pronunciation.pinyin}</div>`;
    html += `<ul style="margin:6px 0 0 18px;padding:0">${items.map((t:any)=>`<li>${t}</li>`).join('')}</ul>`;
    if (label) html += `<div class="label">${label}</div>`;
  } else {
    html += header;
    if (pronunciation?.pinyin) html += `<div class="label">${pronunciation.pinyin}</div>`;
    if (label) html += `<div class="label">${label}</div>`;
    html += `<div style="color:#999">No dictionary entry found</div>`;
  }
  popover.innerHTML = html;
  popover.style.display = 'block';
}

function renderText() {
  localStorage.setItem('arcadia_text', inputEl.value);
  outEl.innerHTML = '';
  clearPopover();
  const lang = srcSel.value;
  if (lang.startsWith('zh')) {
    // Server-side parse with char-level pinyin
    parse(lang, inputEl.value).then(data => {
      for (const t of data.tokens) {
        if (t.is_word && t.text && t.text.trim()) {
          const span = document.createElement('span');
          span.className = 'token';
          span.setAttribute('data-text', t.text);
          if (t.chars?.length) {
            for (const ch of t.chars) {
              const cspan = document.createElement('span');
              cspan.className = 'char';
              cspan.textContent = ch.ch;
              if (ch.pinyin) cspan.title = `${ch.pinyin}`;
              span.appendChild(cspan);
            }
          } else {
            span.textContent = t.text;
          }
          outEl.appendChild(span);
        } else {
          outEl.appendChild(document.createTextNode(t.text));
        }
      }
    }).catch(() => {
      // fallback to client tokenization
      const toks = tokenize(inputEl.value);
      for (const tok of toks) {
        if (tok.isWord) {
          const span = document.createElement('span');
          span.className = 'token';
          span.textContent = tok.t;
          span.setAttribute('data-text', tok.t);
          outEl.appendChild(span);
        } else {
          outEl.appendChild(document.createTextNode(tok.t));
        }
      }
    });
  } else {
    const toks = tokenize(inputEl.value);
    for (const tok of toks) {
      if (tok.isWord) {
        const span = document.createElement('span');
        span.className = 'token';
        span.textContent = tok.t;
        span.setAttribute('data-text', tok.t);
        outEl.appendChild(span);
      } else {
        outEl.appendChild(document.createTextNode(tok.t));
      }
    }
  }
}

// Init
inputEl.value = localStorage.getItem('arcadia_text') || inputEl.value;
renderBtn.addEventListener('click', renderText);
// Delegate clicks from container to token spans
outEl.addEventListener('click', async (e) => {
  const target = e.target as HTMLElement;
  const tokenEl = target.closest('.token') as HTMLElement | null;
  if (!tokenEl) return;
  e.stopPropagation();
  const surface = tokenEl.getAttribute('data-text') || tokenEl.textContent || '';
  const rect = tokenEl.getBoundingClientRect();
  try {
    const data = await lookup(srcSel.value, tgtSel.value, surface);
    showPopover(rect.left + window.scrollX, rect.bottom + window.scrollY + 6, data);
  } catch {
    showPopover(rect.left + window.scrollX, rect.bottom + window.scrollY + 6, { mode:'analysis', lemma:surface, translations:[], label:'error' });
  }
});
document.addEventListener('click', (e) => { if (!popover.contains(e.target as Node)) clearPopover(); });

// Hover tooltip for per-character pinyin (zh only)
outEl.addEventListener('mouseover', (e) => {
  const target = e.target as HTMLElement;
  if (!target.classList.contains('char')) return;
  const t = target.getAttribute('title') || '';
  if (!t) return;
  hoverTip.textContent = t;
  hoverTip.style.display = 'block';
});
outEl.addEventListener('mousemove', (e) => {
  if (hoverTip.style.display !== 'block') return;
  hoverTip.style.left = `${e.pageX + 10}px`;
  hoverTip.style.top = `${e.pageY + 10}px`;
});
outEl.addEventListener('mouseout', (e) => {
  const target = e.target as HTMLElement;
  if (!target.classList.contains('char')) return;
  hoverTip.style.display = 'none';
});
renderText();
