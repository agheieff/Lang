type Token = { t: string; isWord: boolean };

const inputEl = document.getElementById('input') as HTMLTextAreaElement;
const outEl = document.getElementById('output') as HTMLDivElement;
const renderBtn = document.getElementById('render') as HTMLButtonElement;
const srcSel = document.getElementById('src') as HTMLSelectElement;
const tgtSel = document.getElementById('tgt') as HTMLSelectElement;
const popover = document.getElementById('popover') as HTMLDivElement;
const hoverTip = document.getElementById('hover-tip') as HTMLDivElement;
const tierSel = document.getElementById('tier') as HTMLSelectElement;
const emailEl = document.getElementById('email') as HTMLInputElement;
const pwdEl = document.getElementById('password') as HTMLInputElement;
const loginBtn = document.getElementById('login') as HTMLButtonElement;
const registerBtn = document.getElementById('register') as HTMLButtonElement;
const logoutBtn = document.getElementById('logout') as HTMLButtonElement;
const pinyinSel = document.getElementById('pinyin-style') as HTMLSelectElement;
const saveSettingsBtn = document.getElementById('save-settings') as HTMLButtonElement;
const profileLangInput = document.getElementById('profile-lang') as HTMLInputElement;
const addProfileBtn = document.getElementById('add-profile') as HTMLButtonElement;
const profilesDiv = document.getElementById('profiles') as HTMLDivElement;
const genBtn = document.getElementById('gen') as HTMLButtonElement;
const msgEl = document.getElementById('msg') as HTMLDivElement;

// Auth tokens (simple localStorage for demo)
function getAccessToken() { return localStorage.getItem('arcadia_access') || ''; }
function getRefreshToken() { return localStorage.getItem('arcadia_refresh') || ''; }
function setTokens(a: string, r: string) { localStorage.setItem('arcadia_access', a); localStorage.setItem('arcadia_refresh', r); }

async function api(path: string, opts: RequestInit = {}) {
  const headers: any = { 'Content-Type': 'application/json', ...(opts.headers||{}) };
  const token = getAccessToken();
  if (token) headers['Authorization'] = `Bearer ${token}`;
  let res = await fetch(path, { ...opts, headers });
  if (res.status === 401 && getRefreshToken()) {
    // try refresh
    const rr = await fetch('/auth/refresh', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ refresh_token: getRefreshToken() }) });
    if (rr.ok) {
      const data = await rr.json();
      if (data.access_token) setTokens(data.access_token, data.refresh_token || getRefreshToken());
      headers['Authorization'] = `Bearer ${getAccessToken()}`;
      res = await fetch(path, { ...opts, headers });
    }
  }
  return res;
}

function showMsg(ok: boolean, text: string) {
  if (!msgEl) return;
  msgEl.className = 'msg ' + (ok ? 'ok' : 'err');
  msgEl.textContent = text;
}

function tokenize(text: string): Token[] {
  const re = /([\p{L}\p{M}]+)|([^\p{L}\p{M}]+)/gu;
  const out: Token[] = [];
  for (const m of text.matchAll(re) as any) out.push({ t: m[0], isWord: !!m[1] });
  return out;
}

async function lookup(source_lang: string, target_lang: string, surface: string) {
  const res = await api('/api/lookup', { method: 'POST', body: JSON.stringify({ source_lang, target_lang, surface }) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function parse(lang: string, text: string) {
  const res = await api('/api/parse', { method: 'POST', body: JSON.stringify({ lang, text }) });
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
  // Collect exposures after a small debounce
  const exposures: { lemma?: string; surface?: string }[] = [];
  if (lang.startsWith('zh')) {
    // Server-side parse with char-level pinyin
    parse(lang, inputEl.value).then(data => {
      const seen = new Set<string>();
      for (const t of data.tokens) {
        if (t.is_word && t.text && t.text.trim()) {
          if (!seen.has(t.text)) { exposures.push({ surface: t.text }); seen.add(t.text); }
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
      // fire exposures (surface only; server resolves lemmas)
      sendExposures(lang, exposures);
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
    const seen = new Set<string>();
    for (const tok of toks) {
      if (tok.isWord) {
        const span = document.createElement('span');
        span.className = 'token';
        span.textContent = tok.t;
        span.setAttribute('data-text', tok.t);
        outEl.appendChild(span);
        if (!seen.has(tok.t)) { exposures.push({ surface: tok.t }); seen.add(tok.t); }
      } else {
        outEl.appendChild(document.createTextNode(tok.t));
      }
    }
    sendExposures(lang, exposures);
  }
}

// Init
inputEl.value = localStorage.getItem('arcadia_text') || inputEl.value;
renderBtn.addEventListener('click', renderText);
// Load tiers into selector
async function loadTiers() {
  const res = await api('/tiers');
  if (!res.ok) return;
  const tiers = await res.json();
  tierSel.innerHTML = '';
  for (const t of tiers) {
    const opt = document.createElement('option');
    opt.value = t.name; opt.textContent = t.name;
    tierSel.appendChild(opt);
  }
  // Fetch current user's tier if logged in
  const mt = await api('/me/tier');
  if (mt.ok) {
    const cur = await mt.json();
    tierSel.value = cur.name;
  }
}
tierSel?.addEventListener('change', async () => {
  const name = tierSel.value;
  const res = await api('/me/tier', { method: 'POST', body: JSON.stringify({ name }) });
  if (!res.ok) {
    alert('Failed to set tier');
  }
});
loadTiers();

async function refreshProfiles() {
  const res = await api('/me/profiles');
  if (!res.ok) { profilesDiv.textContent = 'Login to manage profiles'; return; }
  const items = await res.json();
  profilesDiv.innerHTML = '<strong>Your profiles:</strong> ' + items.map((p:any)=>`${p.lang}`).join(', ');
}
refreshProfiles();

async function auth(path: string) {
  const body = { email: emailEl.value, password: pwdEl.value };
  const res = await fetch(path, { method: 'POST', headers: { 'Content-Type':'application/json' }, body: JSON.stringify(body) });
  const txt = await res.text();
  if (!res.ok) { try { const j = JSON.parse(txt); showMsg(false, j.detail || 'Auth failed'); } catch { showMsg(false, txt || 'Auth failed'); } return; }
  try {
    const data = JSON.parse(txt);
    if (data.access_token) setTokens(data.access_token, data.refresh_token || '');
  } catch { /* ignore */ }
  showMsg(true, path.includes('register') ? 'Registered and logged in' : 'Logged in');
  await loadTiers();
  await refreshProfiles();
}
loginBtn.onclick = () => auth('/auth/login');
registerBtn.onclick = () => auth('/auth/register');
logoutBtn.onclick = () => { setTokens('', ''); showMsg(true, 'Logged out'); };

saveSettingsBtn.onclick = async () => {
  const lang = srcSel.value;
  const style = pinyinSel.value === 'tone' ? 'tone' : 'number';
  const settings = { zh_pinyin_style: style };
  const res = await api('/me/profile', { method: 'POST', body: JSON.stringify({ lang, settings }) });
  if (!res.ok) alert('Save failed');
};

addProfileBtn.onclick = async () => {
  const lang = profileLangInput.value.trim();
  if (!lang) return;
  const res = await api('/me/profile', { method: 'POST', body: JSON.stringify({ lang }) });
  if (!res.ok) { alert('Failed'); return; }
  profileLangInput.value = '';
  await refreshProfiles();
};
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

async function sendExposures(lang: string, items: { lemma?: string; surface?: string }[]) {
  if (!items.length) return;
  try {
    const res = await api('/srs/event/exposures', { method: 'POST', body: JSON.stringify({ lang, items }) });
    // ignore failures silently for now
  } catch {}
}

// Generate text via LLM and render it immediately
genBtn.addEventListener('click', async () => {
  const lang = srcSel.value;
  // Length hint per language; could be expanded later
  const length = lang.startsWith('zh') ? 300 : 180;
  try {
    const res = await api('/gen/reading', { method: 'POST', body: JSON.stringify({ lang, length }) });
    if (!res.ok) { showMsg(false, 'Generation failed'); return; }
    const data = await res.json();
    inputEl.value = data.text || '';
    renderText();
    showMsg(true, 'Generated text inserted');
  } catch {
    showMsg(false, 'Generation failed');
  }
});
