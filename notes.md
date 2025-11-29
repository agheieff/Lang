# Development Notes

Issues we struggled with and their solutions.

---

## 1. HTMX responses render empty despite correct server response

**Symptom**: HTMX `hx-get` requests return valid HTML (visible in Network tab), `htmx:afterSwap` fires, but target element remains empty.

**Root cause**: Global `hx-select` on `<body>` filters responses.

In `base.html`, we set SPA-like navigation:
```javascript
document.body.setAttribute('hx-select', '#content');
```

This tells HTMX: "from ANY response, only extract elements with `id='content'`". If your endpoint returns `<div id="something-else">`, HTMX extracts nothing → empty swap.

**Why `hx-disinherit="*"` didn't help**: The body attributes are set via JavaScript after DOMContentLoaded, not in HTML. `hx-disinherit` may not fully prevent inheritance of dynamically-added attributes.

**Solution**: Add `hx-select="unset"` to elements that fetch non-`#content` responses:

```html
<div id="current-reading"
     hx-get="/reading/current"
     hx-trigger="load delay:200ms"
     hx-swap="innerHTML"
     hx-target="this"
     hx-select="unset"
     hx-disinherit="*">
```

**Affected areas**: Any HTMX request where the response doesn't contain `id="content"`:
- `/reading/current` → returns `#reading-block`
- `/settings/tier` → returns tier HTML
- `/settings/topics` → returns topics HTML
- Model management endpoints
