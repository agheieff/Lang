// Arcadia Lang - Context Menu Module (vanilla JS, no deps)
// Exposes window.arcContextMenu with register/unregisterAll APIs.
(function(){
  'use strict';

  const state = {
    rules: [],
    menu: null,
    list: null,
    isOpen: false,
    lastEvent: null,
    hideOnNextUp: false,
  };

  function ensureMenu(){
    if(state.menu) return state.menu;
    const menu = document.createElement('div');
    menu.className = 'arc-context-menu';
    menu.id = 'arc-context-menu';
    menu.setAttribute('role','menu');
    menu.setAttribute('aria-hidden','true');
    menu.tabIndex = -1;
    const ul = document.createElement('ul');
    ul.className = 'arc-menu-list';
    menu.appendChild(ul);
    document.body.appendChild(menu);
    state.menu = menu;
    state.list = ul;

    // Global hide handlers
    window.addEventListener('scroll', hide, true);
    window.addEventListener('resize', hide, true);
    document.addEventListener('click', (e)=>{ if(state.isOpen){ hide(); } }, true);
    document.addEventListener('keydown', onKeyNav, true);
    return menu;
  }

  function clearMenu(){
    const ul = state.list; if(!ul) return;
    while(ul.firstChild) ul.removeChild(ul.firstChild);
  }

  function createItemEl(item, ctx){
    if(item == null) return null;
    if(item.divider || item.separator === true) {
      const sep = document.createElement('div');
      sep.className = 'arc-menu-sep';
      sep.setAttribute('role','separator');
      return sep;
    }
    const btn = document.createElement('button');
    btn.className = 'arc-menu-item';
    btn.setAttribute('role','menuitem');
    btn.type = 'button';
    btn.tabIndex = -1;
    if(item.disabled) btn.setAttribute('disabled','');
    const label = document.createElement('span');
    label.className = 'arc-menu-label';
    label.textContent = String(item.label || '');
    btn.appendChild(label);

    // HTMX wiring (if provided)
    const hx = item.hx || {};
    const hasHx = typeof window.htmx !== 'undefined' && (hx.get || hx.post || hx.delete || item.hxGet || item.hxPost || item.hxDelete);
    if(item.hxGet) btn.setAttribute('hx-get', item.hxGet);
    if(item.hxPost) btn.setAttribute('hx-post', item.hxPost);
    if(item.hxDelete) btn.setAttribute('hx-delete', item.hxDelete);
    if(hx.get) btn.setAttribute('hx-get', hx.get);
    if(hx.post) btn.setAttribute('hx-post', hx.post);
    if(hx.delete) btn.setAttribute('hx-delete', hx.delete);
    if(item.hxTarget || hx.target) btn.setAttribute('hx-target', item.hxTarget || hx.target);
    if(item.hxSwap || hx.swap) btn.setAttribute('hx-swap', item.hxSwap || hx.swap);
    if(item.hxVals || hx.vals){ try{ btn.setAttribute('hx-vals', JSON.stringify(item.hxVals || hx.vals)); }catch(_e){} }
    if(item.hxExt || hx.ext) btn.setAttribute('hx-ext', item.hxExt || hx.ext);
    if(item.hxConfirm || hx.confirm) btn.setAttribute('hx-confirm', item.hxConfirm || hx.confirm);

    btn.addEventListener('click', function(ev){
      ev.preventDefault(); ev.stopPropagation();
      if(btn.hasAttribute('disabled')) return;
      hide();
      try{
        if(item.onClick && typeof item.onClick === 'function'){
          return item.onClick(ctx);
        }
        if(hasHx){
          // Let HTMX handle the click
          if(typeof window.htmx !== 'undefined'){
            // Ensure htmx sees this node
            window.htmx.process(btn);
          }
          btn.click();
        }
      }catch(_e){}
    }, { once: true });

    return btn;
  }

  function viewportConstrain(x, y, menu){
    const vw = window.innerWidth, vh = window.innerHeight;
    const rect = menu.getBoundingClientRect();
    let nx = x, ny = y;
    if(nx + rect.width > vw - 6) nx = Math.max(6, vw - rect.width - 6);
    if(ny + rect.height > vh - 6) ny = Math.max(6, vh - rect.height - 6);
    return { x: nx, y: ny };
  }

  function showAt(x, y){
    const menu = ensureMenu();
    menu.style.visibility = 'hidden';
    menu.style.left = '0px';
    menu.style.top = '0px';
    menu.setAttribute('data-open','true');
    menu.removeAttribute('aria-hidden');
    // Force layout, then position
    const pos = viewportConstrain(x, y, menu);
    menu.style.left = pos.x + 'px';
    menu.style.top = pos.y + 'px';
    menu.style.visibility = 'visible';
    state.isOpen = true;
    // Focus first item
    const first = menu.querySelector('.arc-menu-item');
    if(first){ try{ first.focus(); first.setAttribute('aria-selected','true'); }catch(_e){} }
  }

  function hide(){
    if(!state.menu || !state.isOpen) return;
    state.menu.setAttribute('aria-hidden','true');
    state.menu.removeAttribute('data-open');
    state.isOpen = false;
    state.lastEvent = null;
    clearMenu();
  }

  function onKeyNav(e){
    if(!state.isOpen || !state.menu) return;
    const items = Array.from(state.menu.querySelectorAll('.arc-menu-item'));
    if(!items.length) return;
    const current = document.activeElement;
    const idx = Math.max(0, items.indexOf(current));
    if(e.key === 'Escape') { e.preventDefault(); hide(); return; }
    if(e.key === 'ArrowDown'){
      e.preventDefault();
      const next = items[(idx + 1) % items.length];
      try{ current && current.setAttribute('aria-selected','false'); }catch(_e){}
      if(next){ next.focus(); next.setAttribute('aria-selected','true'); }
    } else if(e.key === 'ArrowUp'){
      e.preventDefault();
      const prev = items[(idx - 1 + items.length) % items.length];
      try{ current && current.setAttribute('aria-selected','false'); }catch(_e){}
      if(prev){ prev.focus(); prev.setAttribute('aria-selected','true'); }
    } else if(e.key === 'Enter'){
      e.preventDefault();
      if(current && typeof current.click === 'function') current.click();
    }
  }

  function closestMatch(target, selector){
    if(!selector) return null;
    try{ return target.closest(selector); }catch(_e){ return null; }
  }

  function getSelectionText(){
    try{ return String(window.getSelection && window.getSelection().toString() || ''); }catch(_e){ return ''; }
  }

  function onContextMenu(ev){
    // Determine if any rule applies
    let matched = null, scopeEl = null;
    for(let i=state.rules.length-1;i>=0;i--){ // last-registered wins
      const r = state.rules[i];
      const el = closestMatch(ev.target, r.selector);
      if(!el) continue;
      const ctx = {
        event: ev,
        target: el,
        originalTarget: ev.target,
        selection: getSelectionText(),
        isInput: el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable === true,
      };
      try{
        if(r.when && r.when(ctx) === false) continue;
        const items = (typeof r.items === 'function') ? r.items(ctx) : (Array.isArray(r.items) ? r.items : []);
        const arr = Array.isArray(items) ? items.filter(Boolean) : [];
        if(arr.length){ matched = { rule: r, items: arr, ctx }; scopeEl = el; break; }
      }catch(_e){ continue; }
    }
    if(!matched){ return; } // let browser menu show
    // Intercept only when we will show
    ev.preventDefault(); ev.stopPropagation();
    state.lastEvent = ev;
    ensureMenu();
    clearMenu();
    // Build items
    for(const it of matched.items){
      const el = createItemEl(it, matched.ctx);
      if(!el) continue;
      state.list.appendChild(el);
    }
    // If nothing meaningful, fall back to default
    if(!state.list.children.length){ return; }
    const x = ev.clientX, y = ev.clientY;
    showAt(x, y);
  }

  function register(rule){
    if(!rule || !rule.selector) return ()=>{};
    state.rules.push(rule);
    return function unregister(){
      const idx = state.rules.indexOf(rule);
      if(idx >= 0) state.rules.splice(idx,1);
    };
  }

  function unregisterAll(){
    state.rules.length = 0;
  }

  function init(){
    if(typeof document === 'undefined') return;
    ensureMenu();
    // Register handler once
    if(!document.__arcCtxMenuBound){
      document.addEventListener('contextmenu', onContextMenu, true);
      document.__arcCtxMenuBound = true;
    }
  }

  // Expose API
  const api = { init, register, unregisterAll };
  Object.defineProperty(window, 'arcContextMenu', { value: api, writable: false, configurable: false });

  // Auto-init after DOM ready
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
