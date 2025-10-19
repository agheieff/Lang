# Language pipeline (MVP)

This app turns pasted text into clickable tokens that show analysis and translations.

What happens by language
- Spanish (es):
  - Tokenization: client regex preserves whitespace/punctuation.
  - Analysis: FastAPI uses spaCy (if available) or heuristics to return lemma (infinitive), POS, and a compact morph label (e.g., `3sg ind pret`).
  - Translation: StarDict/FreeDict dictionaries from `data/dicts/es-en/` (or `ARCADIA_DICT_ROOT/es-en`) via `pystardict`.

- Chinese (zh, zh-Hans, zh-Hant):
  - Tokenization: server `/api/parse` uses jieba with a fallback “forward maximum matching” over CC‑CEDICT to segment contiguous Han text; returns tokens plus per‑character pinyin.
  - Analysis: lemma is normalized to Simplified (OpenCC); POS is a coarse mapping from jieba flags; response includes `script` and full‑token pinyin.
  - Translation: CC‑CEDICT (`data/dicts/zh-en/cedict_ts.u8`) via a `CedictProvider`, with StarDict as secondary.

Frontend behavior
- Non‑zh: client tokenizes; clicking a word calls `/api/lookup` and shows lemma, POS, morph, and translations.
- zh: client calls `/api/parse`; renders token spans containing per‑character spans. Hover shows per‑char pinyin; clicking the token calls `/api/lookup` and shows translations and full‑token pinyin (diacritics by default).

Add dictionaries
- StarDict: put matching `.ifo/.idx/.dict(.dz)` files under `data/dicts/{src}-{tgt}/`.
- CC‑CEDICT: place `cedict_ts.u8` under `data/dicts/zh-en/`.

Notes
- Lemma normalization for Chinese is Simplified; the original script is returned in `script`.
- Dictionary provider chain is pluggable; Chinese prefers CEDICT first.
