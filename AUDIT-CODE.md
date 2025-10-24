 # Core Domain Audit — Arcadia Lang
 
 Scope: Tokenization, word analysis, dictionary lookup, SRS events/selection, and LLM reading generation. Auth/UI omitted per brief.
 
 ## 1) Architecture Overview (core pipeline)
 
 High-level flow from user input to learning loop:
 
 1. Tokenization
    - Entrypoint: `POST /api/parse` in `server/main.py:632`.
    - Dispatch: `Lang/tokenize/registry.py` maps `lang -> Tokenizer`.
      - Latin scripts: `LatinTokenizer.tokenize` (`Lang/tokenize/latin.py`) segments by Unicode letter classes while preserving separators.
      - Chinese: `ZhTokenizer.tokenize` (`Lang/tokenize/zh.py:93`) uses `jieba.tokenize`, with fallback Forward-Maximum-Matching over CC‑CEDICT to repair over/under-segmentation.
    - For zh, `parse` also attaches per-character pinyin via `pypinyin`.
 
 2. Word analysis (lemma/POS/morph)
    - Entrypoint: `POST /api/lookup` in `server/main.py:304`.
    - Dispatch: `Lang/parsing/registry.py` provides `ENGINES`.
      - Spanish: `analyze_word_es` (`Lang/parsing/es.py:164`) attempts spaCy model, falls back to `simplemma`, then heuristics for lemma/POS/morph.
      - Chinese: `analyze_word_zh` (`Lang/parsing/zh.py:33`) normalizes lemma to Simplified via OpenCC, coarse POS from jieba flags, returns script and pinyin.
 
 3. Dictionary lookup
    - Provider chain: `DictionaryProviderChain` (`Lang/parsing/dicts/provider.py`) with in-memory LRU of translation results.
    - Providers:
      - `CedictProvider` (`Lang/parsing/dicts/cedict.py:47`): loads CC‑CEDICT into memory once; per-call OpenCC fallback for script variants.
      - `StarDictProvider` (`Lang/parsing/dicts/provider.py:15`): StarDict via `pystardict`; currently opens dictionary files per request.
    - Usage: `/api/lookup` tries lemma first; for zh, falls back to surface if lemma miss (`server/main.py:318–322`).
 
 4. SRS event logging and selection
    - Events:
      - Click: `_srs_click` (`server/main.py:282`) and `POST /srs/event/click` (`server/main.py:918`).
      - Exposure: `_srs_exposure` (`server/main.py:814`) and `POST /srs/event/exposures` (`server/main.py:843`).
      - Non-lookup confirmations: `_srs_nonlookup` + `POST /srs/event/nonlookup` (`server/main.py:888`).
    - Lexeme resolution and variants (Hans/Hant): `_resolve_lexeme` (`server/main.py:209`) creates or finds `Lexeme` and `LexemeVariant` rows; ensures zh canonical lemma is Simplified and captures variants.
    - User association: `_get_or_create_userlexeme` links user/profile to lexeme.
    - Selection: `urgent_words_detailed` (`server/llm.py:208`) scores known words by stability/click-propensity and importance; mixes in new words from `LexemeInfo` near level/frequency.
 
 5. LLM reading generation
    - Entrypoint: `POST /gen/reading` (`server/main.py:695`).
    - Word selection: `pick_words` (`server/llm.py:291`) -> forms resolved via variants (`_variant_form_for_lang`).
    - Prompt: `build_reading_prompt` (`server/llm.py`) and completion via `chat_complete` (OpenRouter or LM Studio).
    - Persistence: `ReadingText` and `GenerationLog` rows inserted.
 
 ---
 
 ## 2) Prioritized Recommendations (High → Low)
 
 Each item lists the issue, impact, concrete change, and effort.
 
 ### [High] Heavy per-request model/library initialization (spaCy, OpenCC, StarDict, pypinyin)
 • Where:
   - `Lang/parsing/es.py:164` loads spaCy model on every call.
   - `Lang/parsing/zh.py:33` constructs OpenCC instances per call; `server/main.py:_resolve_lexeme:209` also constructs OpenCC each call.
   - `Lang/parsing/dicts/cedict.py:59` constructs OpenCC in `translations` on misses.
   - `server/main.py:662–666` calls `pypinyin.lazy_pinyin` twice per character.
   - `Lang/parsing/dicts/provider.py:15` opens StarDict dictionaries per lookup.
 • Impact: Large latency spikes under load; unnecessary CPU; inconsistent behavior if models fail mid-request; degrades throughput and user-perceived snappiness.
 • Fix:
   - Introduce module-level lazy singletons for spaCy (es), OpenCC (zh), and reuse them everywhere.
   - Cache StarDict `Dictionary` handles per base path in `StarDictProvider` and cache discovered `.ifo` bases.
   - Compute pinyin per token string (vectorized), not per character invocation.
 • Effort: 45–75 minutes. See diffs in section 3.
 
 ### [High] Transaction boundaries inside helpers causing partial commits and batching loss
 • Where:
   - `_resolve_lexeme` (`server/main.py:209`) calls `db.commit()` and `db.refresh()` internally.
   - `_get_or_create_userlexeme` commits per creation.
 • Impact: Breaks atomicity for batch endpoints (`/srs/event/exposures`), increases transaction count, and risks partially committed state (lexeme created without corresponding event) on mid-batch failures.
 • Fix:
   - Restrict helpers to `db.flush()` only (to obtain IDs) and let the endpoint (or service layer) own a single commit per request.
   - Add defensive `IntegrityError` handling for unique constraints (`Lexeme`, `LexemeVariant`, `UserLexeme`) with re-query on conflict.
 • Effort: 30–60 minutes. See minimal diff below.
 
 ### [High] Determinism for lemma/script and lookup fallback
 • Where: `/api/lookup` (`server/main.py:304`) uses lemma-first; for zh only, falls back to surface.
 • Impact: For non-zh, lemma may fail to exist in StarDict while surface does (inflected forms sometimes present). Current behavior drops translations entirely in such cases.
 • Fix:
   - After lemma lookup miss, also attempt surface for all languages (bounded by provider chain cache so low cost). Keep zh special-casing for script variants.
   - Centralize lemma normalization rules per language (e.g., always lower-case for es lemma; already Simplified for zh) to avoid cache fragmentation.
 • Effort: 20–30 minutes.
 
 ### [Medium] Event data completeness: missing context hash on click
 • Where: `POST /api/lookup` ignores `LookupRequest.context` in `_srs_click` call (`server/main.py:341` uses `context_hash=None`).
 • Impact: Diversity metric (`distinct_texts`) undercounts; selection scoring in `urgent_words_detailed` loses signal.
 • Fix: Pass `_hash_context(req.context)` to `_srs_click` in `/api/lookup`.
 • Effort: 5 minutes.
 
 ### [Medium] Query/index tuning hot paths (confirm coverage)
 • Observations:
   - Hot lookups use columns covered by unique constraints: `Lexeme(lang, lemma, pos)`, `LexemeVariant(script, form)`, `UserLexeme(user_id, profile_id, lexeme_id)` — these create composite unique indexes in SQLite.
   - Frequent non-unique filters: `Profile(user_id, lang)` guarded by `uq_profile_user_lang` (OK), `WordEvent(profile_id)`, `UserLexeme(profile_id)` (indexes exist on individual columns).
 • Suggestions:
   - Add an index to `word_events(event_type)` only if querying by type grows (currently not used in endpoints).
   - Consider a covering composite on `word_events (user_id, profile_id, lexeme_id)` if analytics endpoints are added. Current endpoints are OK.
 • Effort: 15–30 minutes (as-needed).
 
 ### [Medium] Modularity: extract domain services from FastAPI routes
 • Where: `server/main.py` contains SRS logic, lexeme resolution, and preference utilities alongside HTTP wiring.
 • Impact: Harder testing/reuse; larger file; mixed concerns.
 • Fix: Extract into small modules (no new deps):
   - `server/domain/lexeme_service.py`: `_resolve_lexeme`, `_get_or_create_userlexeme` (+ conflict handling).
   - `server/domain/srs_service.py`: `_srs_click`, `_srs_exposure`, `_srs_nonlookup`.
   - Route layer becomes thin coordinators.
 • Effort: 1–2 hours.
 
 ### [Low] Minor code issues and polish
 • `Lang/tokenize/zh.py` has unreachable `return out` after an earlier return.
 • `Lang/parsing/morph_format.py` logic is fine; consider small unit tests for common patterns.
 • `server/llm.py urgent_words_detailed` fetches up to 5000 candidates — acceptable for SQLite; revisit when data grows.
 
 ---
 
 ## 3) Quick wins (<1h) vs. longer-term refactors
 
 Quick wins (do these first):
 1) Cache heavy libs/models and StarDict handles (spaCy, OpenCC, StarDict); batch pinyin calls.
 2) Move commits out of `_resolve_lexeme` / `_get_or_create_userlexeme` and rely on outer commit.
 3) Pass `context` hash in `/api/lookup` SRS click logging.
 
 Longer-term:
 4) Extract domain services from FastAPI routes; add conflict-safe upserts; add focused tests.
 5) Optional SQL indexing adjustments if analytics endpoints expand.
 
 ### Minimal example diffs (top items)
 
 1) Lazy singletons for spaCy/OpenCC + batch pinyin
 
 ```diff
 *** file: Lang/parsing/es.py
 @@
 -def analyze_word_es(surface: str, context: Optional[str] = None) -> Dict[str, Any]:
 +_ES_NLP = None  # lazy global
 +
 +def _get_es_nlp():
 +    global _ES_NLP
 +    if _ES_NLP is not None:
 +        return _ES_NLP
 +    try:
 +        import spacy  # type: ignore
 +        for model in ("es_core_news_sm", "es_core_news_md", "es_core_news_lg"):
 +            try:
 +                _ES_NLP = spacy.load(model)
 +                break
 +            except Exception:
 +                continue
 +    except Exception:
 +        _ES_NLP = None
 +    return _ES_NLP
 +
 +def analyze_word_es(surface: str, context: Optional[str] = None) -> Dict[str, Any]:
 @@
 -    try:
 -        import spacy  # type: ignore
 -        for model in ("es_core_news_sm", "es_core_news_md", "es_core_news_lg"):
 -            try:
 -                nlp = spacy.load(model)
 -                break
 -            except Exception:
 -                nlp = None  # type: ignore
 -        if nlp is not None:  # type: ignore
 -            doc = nlp(surface)
 +    try:
 +        nlp = _get_es_nlp()
 +        if nlp is not None:
 +            doc = nlp(surface)  # type: ignore[call-arg]
              token = doc[0] if len(doc) else None
              if token:
                  morph: Dict[str, str] = {}
 ```
 
 ```diff
 *** file: Lang/parsing/zh.py
 @@
 -def analyze_word_zh(surface: str, context: Optional[str] = None) -> Dict[str, Any]:
 +_CC_T2S = None
 +_CC_S2T = None
 +
 +def _cc():
 +    global _CC_T2S, _CC_S2T
 +    if _CC_T2S is None or _CC_S2T is None:
 +        try:
 +            from opencc import OpenCC  # type: ignore
 +            _CC_T2S = OpenCC("t2s")
 +            _CC_S2T = OpenCC("s2t")
 +        except Exception:
 +            _CC_T2S = _CC_S2T = None
 +    return _CC_T2S, _CC_S2T
 +
 +def analyze_word_zh(surface: str, context: Optional[str] = None) -> Dict[str, Any]:
 @@
 -    try:
 -        from opencc import OpenCC  # type: ignore
 -        cc_t2s = OpenCC("t2s")
 -        cc_s2t = OpenCC("s2t")
 -        simp = cc_t2s.convert(s)
 -        trad = cc_s2t.convert(s)
 +    try:
 +        cc_t2s, cc_s2t = _cc()
 +        simp = cc_t2s.convert(s) if cc_t2s else s
 +        trad = cc_s2t.convert(s) if cc_s2t else s
          script = "Hant" if s == trad and s != simp else "Hans"
      except Exception:
          simp = s
          script = None
 ```
 
 ```diff
 *** file: server/main.py
 @@ def parse(req: ParseRequest) -> Dict[str, Any]:
 -        if req.lang.startswith("zh"):
 -            # per-character pinyin
 -            try:
 -                from pypinyin import lazy_pinyin, Style  # type: ignore
 -                chars = []
 -                for i, ch in enumerate(w.text):
 -                    p_mark = lazy_pinyin(ch, style=Style.TONE)
 -                    p_num = lazy_pinyin(ch, style=Style.TONE3)
 -                    chars.append({
 -                        "ch": ch,
 -                        "start": w.start + i,
 -                        "end": w.start + i + 1,
 -                        "pinyin": p_mark[0] if p_mark else None,
 -                        "pinyin_num": p_num[0] if p_num else None,
 -                    })
 -                entry["chars"] = chars
 -            except Exception:
 -                pass
 +        if req.lang.startswith("zh"):
 +            # per-character pinyin (batched per token)
 +            try:
 +                from pypinyin import lazy_pinyin, Style  # type: ignore
 +                p_mark = lazy_pinyin(w.text, style=Style.TONE)
 +                p_num = lazy_pinyin(w.text, style=Style.TONE3)
 +                chars = []
 +                for i, ch in enumerate(w.text):
 +                    chars.append({
 +                        "ch": ch,
 +                        "start": w.start + i,
 +                        "end": w.start + i + 1,
 +                        "pinyin": p_mark[i] if i < len(p_mark) else None,
 +                        "pinyin_num": p_num[i] if i < len(p_num) else None,
 +                    })
 +                entry["chars"] = chars
 +            except Exception:
 +                pass
 ```
 
 2) Cache StarDict dictionary handles and discovered bases
 
 ```diff
 *** file: Lang/parsing/dicts/provider.py
 @@ class StarDictProvider(DictionaryProvider):
      def __init__(self, root: Optional[Path] = None) -> None:
 @@
          self._ok = False
 +        self._handles: dict[str, object] = {}
 +        self._bases_cache: dict[str, list[Path]] = {}
 @@
      def _dict_paths(self, src: str, tgt: str) -> List[Path]:
 -        pair = f"{src}-{tgt}"
 -        d = self.root / pair
 +        pair = f"{src}-{tgt}"
 +        if pair in self._bases_cache:
 +            return self._bases_cache[pair]
 +        d = self.root / pair
          if not d.exists() or not d.is_dir():
 -            return []
 +            self._bases_cache[pair] = []
 +            return []
 @@
 -        return bases
 +        self._bases_cache[pair] = bases
 +        return bases
 @@
      def translations(self, src: str, tgt: str, lemma: str) -> List[str]:
          if not self._ok:
              return []
          try:
              from pystardict import Dictionary
          except Exception:
              return []
          results: List[str] = []
          for base in self._dict_paths(src, tgt):
 -            try:
 -                d = Dictionary(str(base))
 +            try:
 +                key = str(base)
 +                d = self._handles.get(key)
 +                if d is None:
 +                    d = Dictionary(key)
 +                    self._handles[key] = d
                  if lemma in d:
                      raw = d[lemma]
                      # Prefer extracting <li> entries; fall back to text cleanup
                      items = re.findall(r"<li[^>]*>(.*?)</li>", raw, flags=re.I | re.S)
 ```
 
 3) Move commits out of helpers (keep single commit at endpoints)
 
 ```diff
 *** file: server/main.py
 @@ def _resolve_lexeme(db: Session, lang: str, lemma: str, pos: Optional[str]) -> Lexeme:
 -    db.commit()
 -    db.refresh(lex)
 -    return lex
 +    # Only flush to obtain IDs; outer scope owns commit
 +    db.flush()
 +    return lex
 @@ def _get_or_create_userlexeme(db: Session, user: User, profile: Profile, lexeme: Lexeme) -> UserLexeme:
 -    db.add(ul)
 -    db.commit()
 -    db.refresh(ul)
 -    return ul
 +    db.add(ul)
 +    db.flush()
 +    return ul
 ```
 
 Note: In production, wrap the create paths with `IntegrityError` retry (re-query on conflict) for `Lexeme`, `LexemeVariant`, and `UserLexeme` to be concurrency-safe.
 
 ---
 
 ## 4) Verification checklist (post-change)
 - Warm-start timing for first `es` lookup shows spaCy/model load once; subsequent calls avoid reloads.
 - `zh` lookups do not repeatedly construct OpenCC objects (trace import and object creation count).
 - StarDict lookups reuse cached `Dictionary` handles (instrument construction).
 - Batch `/srs/event/exposures` performs a single DB commit; events and counters match item count, and no partial state on simulated mid-batch failure.
 - `/api/lookup` click now records `context_hash` when provided.
 
 ## 5) Summary of actions and follow-ups
 - Reviewed core files and endpoints; mapped pipeline and hot paths.
 - Identified high-impact caching and transaction-boundary issues; provided concrete diffs for remediation.
 - Blockers/uncertainties: None major; optional conflict-safe upserts need dialect-aware handling if adopted.
 - Follow-ups: Extract domain services for SRS/lexeme; add small unit tests for zh/es analyzers and transaction helpers.
 
