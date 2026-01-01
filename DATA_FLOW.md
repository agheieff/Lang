# Interactive Reading Data Flow

## Complete Pipeline

### 1. LLM Output (CSV Format)
```
word|translation|pos|lemma|pinyin
今天|today|NOUN|今天|jīntiān
天气|weather|NOUN|天气|tiānqì
```

### 2. CSV Parsing (`server/utils/nlp.py:parse_csv_word_translations`)
```python
{
    "surface": "今天",
    "translation": "today",
    "pos": "NOUN",
    "lemma": "今天",
    "pinyin": "jīntiān"
}
```

### 3. Span Computation (`server/utils/nlp.py:compute_word_spans`)
- Finds each word in text **sequentially** (order matters!)
- Returns: `[(start, end)]` for continuous words
- Returns: `[(start1, end1), (start2, end2)]` for non-continuous words (e.g., German `ruf...an`)
- **Critical**: Assumes LLM returns words **in the order they appear in text**

### 4. Database Storage (`server/models.py:ReadingWordGloss`)
```python
ReadingWordGloss(
    text_id=...,
    surface="今天",
    translation="today",
    pos="NOUN",
    lemma="今天",
    pinyin="jīntiān",
    span_start=0,  # First segment only for multi-segment words
    span_end=2,
    grammar={"spans": [...]}  # All spans for multi-segment words
)
```

### 5. Backend to Template (`server/routes/reading.py`)
```python
word_data = [
    {
        "surface": g.surface,
        "lemma": g.lemma,
        "pos": g.pos,
        "translation": g.translation,
        "span_start": g.span_start,
        "span_end": g.span_end,
    }
    for g in word_glosses
]
```

### 6. Frontend Rendering (`server/templates/pages/reading.html`)
- **Validates** spans before using:
  - Checks span bounds (0 ≤ start < end ≤ text length)
  - Verifies extracted text matches surface (fuzzy match)
  - Filters out invalid words
- Renders each word as `<span>` with:
  - `data-word-index`: position in array
  - `data-word-data`: JSON with all word info
  - `data-surface`, `data-lemma`, `data-pos`, `data-translation`: quick access
- Uses `actualText` from the span position (not the surface field)

### 7. User Interaction (`server/static/app.js`)
- **Hover**: Shows tooltip with translation
- **Click**: Tracks interaction via `/reading/word-click`
- **Session**: Stores to localStorage, syncs to server periodically
- **SSE**: Real-time updates for translation availability

## Critical Assumptions

1. **LLM returns words IN ORDER** - `compute_word_spans` is sequential
2. **Spans match text** - Frontend validates this
3. **No word overlaps** - Each character belongs to at most one word span
4. **Unicode consistent** - Text uses same normalization as LLM output

## Potential Failure Points

1. **LLM returns wrong order** → Spans will be misaligned
   - *Mitigation*: Frontend validation filters mismatched words

2. **Multi-segment words** → Only first segment highlighted
   - *Example*: German `anrufen` → `ruf...an` → only `ruf` stored
   - *Status*: Known limitation, acceptable for Chinese

3. **Repeated words** → First occurrence from current_pos used
   - *Example*: "好好" → First "好" highlighted twice
   - *Mitigation*: Sequential scan prevents double-highlighting

4. **Unicode normalization** → Spans might not match
   - *Example*: Combining characters vs precomposed
   - *Mitigation*: Frontend fuzzy matching allows partial matches

## Files Involved

- `server/llm/prompts/word_analysis/*.md` - Prompt format
- `server/utils/nlp.py` - CSV parsing & span computation
- `server/services/content.py` - Storage logic
- `server/models.py` - Database schema
- `server/routes/reading.py` - Backend to template
- `server/templates/pages/reading.html` - Frontend rendering
- `server/static/app.js` - User interaction handling
