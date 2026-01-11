# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Arcadia Lang** is a language learning web application that generates personalized reading practice texts using LLMs. Features include click-to-translate word glosses, sentence translations, and a spaced repetition system (SRS) for vocabulary tracking.

## Development Commands

```bash
# Install dependencies and run development server (port 8000)
./run.sh

# Run with custom port
./run.sh --port 3000

# Run tests
./run_tests.sh

# Run specific test
uv run pytest tests/test_llm_generation.py -v

# Run tests with coverage
uv run pytest --cov=server tests/

# Reset database (keep texts, clear users)
python reset_db.py

# Full database reset (everything)
python reset_db.py --full

# Initialize SRS values for existing lexemes
uv run python server/utils/initialize_srs.py
```

## Required Environment Variables

```bash
# Required
export ARC_LANG_JWT_SECRET="your-secret-key"
export OPENROUTER_API_KEY="your-openrouter-key"

# Optional (with defaults)
export ARC_STARTUP_LANGS="es,zh-CN"          # Languages to pre-generate on startup
export ARC_STARTUP_TEXTS_PER_LANG=2         # Texts per language
export ARC_POOL_SIZE=4                       # Target pool size per profile
export ARC_LLM_TEMPERATURE=0.7              # LLM temperature
```

## Architecture Overview

### Database Design (SQLite: `data/app.db`)

**Global Shared Texts:**
- `reading_texts` - Generated texts shared across users with matching language pairs
- `reading_text_translations` - Sentence/paragraph translations
- `reading_word_glosses` - Word-level translations with span positions
- `text_vocabulary` - Vocabulary index for text matching

**Per-User Data:**
- `accounts` - User authentication with JWT
- `profiles` - Language learning profiles (one per language pair per user)
- `lexemes` - User-specific vocabulary with SRS tracking (keyed by account_id, profile_id, lang, lemma, pos)
- `profile_text_reads` - Reading history tracking
- `profile_text_states` - User's interaction state after reading each text

**Key Design:** Texts are generated once globally but vocabulary learning is tracked per-user. A unique constraint on `(account_id, profile_id, lang, lemma, pos)` prevents duplicate lexemes.

### Text Generation Pipeline

1. **Background Worker** (`server/services/background_worker.py`) runs every 5 minutes:
   - Detects "pool gaps" (profiles with < 3 ready texts)
   - Generates texts to fill gaps (up to 3 concurrent)
   - Retries failed translations (up to 3 attempts)
   - Cleans up old unread texts (> 30 days)

2. **Content Generation** (`server/services/content.py`):
   - `generate_text_content()` - LLM generates text with target comprehension index (CI)
   - `generate_translations()` - Parallel word gloss and sentence translation generation
   - Text status: `is_ready` requires `content + words_complete + sentences_complete`

3. **State Building** (`server/services/text_state_builder.py`):
   - Creates complete JSON state for frontend
   - Includes: content, words with glosses, translations, spans

4. **LLM Client** (`server/llm/client.py`):
   - Multi-provider support (OpenRouter default, local LLM fallback)
   - Exponential backoff retry for rate limits (429, 5xx errors)
   - Model configuration in `server/llm/models.json`

### SRS (Spaced Repetition System)

**SRS Algorithm** (`server/services/srs.py`):
- Tracks per-word: `familiarity` (0-1), `familiarity_variance`, `decay_rate`, `decay_variance`
- Click on word → familiarity -= 0.3 × min(clicks, 3), variance increases
- No click → familiarity += 0.1, variance decreases
- Time decay: `familiarity *= exp(-decay_rate × days_since_seen)`
- Next review calculated from familiarity and variance

**Important:** SRS values are stored per-lexeme (unique per account_id, profile_id, lang, lemma, pos). Same word with different POS (e.g., "一个" DET vs "一个" NUM) are separate lexemes.

**Update Flow:**
- User clicks "Next Text" → `POST /reading/log-text-state`
- `batch_update_lexemes_from_text_state()` processes all words from the text
- Applies time decay once per word, then applies click/no-click signal
- Updates exposures, last_seen_at, calculates next_due_at

### Reading Flow

**Main Reading Page** (`GET /reading` → `routes/reading.py:reading_page()`):
1. Get active profile from `active_profile_id` cookie
2. Call `select_best_text(db, profile)` → `services/recommendation.py`
3. Load word glosses from `ReadingWordGloss` table
4. Render template with interactive word spans
5. Frontend: `trackWordClick()` records clicks in state.words[i].clicks[] array

**Text Selection** (`services/recommendation.py:select_best_text()`):
- Filters texts matching profile's lang/target_lang
- Excludes already-read texts (from `profile_text_reads`)
- Scores by difficulty match, topic preferences, vocabulary overlap
- Returns highest-scoring ready text

**Session Save** (`POST /reading/log-text-state`):
1. Receives complete text state with client-enriched data (clicks, account_id, profile_id, timestamps)
2. Adds words to user's vocabulary (Lexeme table)
3. Updates SRS values from click data
4. Saves state to `ProfileTextState` table

### Frontend State Management

**Text State** (`server/static/reading-state.js`):
- Lives in browser memory during reading session
- Fetched from `/reading/{text_id}/state` on page load
- Client enriches with: account_id, profile_id, loaded_at, clicks[] arrays
- Sent to server on "Next Text" via `saveState()`

**Word Click Tracking** (`server/static/app.js`):
- `trackWordClick()` called when user clicks a word
- Records timestamp in `state.words[i].clicks[]`
- Saves enriched state to localStorage: `arc_text_state_{textId}`
- Also saves to `arc_current_session_{textId}` for session analytics

### Multi-Segment Words

Some languages (Chinese) have discontinuous word spans. The `grammar` field in `ReadingWordGloss` stores all spans:
```json
{
  "surface": "我的",
  "spans": [[0, 2], [10, 12]]
}
```

### Language Support

**Chinese (zh-CN, zh-TW):**
- Jieba tokenization
- Pinyin generation via `pypinyin`
- Prompts in `server/llm/prompts/word_analysis/zh-CN.md`

**Spanish (es):**
- Regex-based tokenization
- Prompts in `server/llm/prompts/word_analysis/es.md`

**Tokenization Rule:** For Chinese compounds like "一个", tokenize as two words ("一" + "个") unless it's a single meaningful unit like "吃饭".

### Background Worker Lifecycle

**Startup** (`server/main.py:lifespan()`):
1. If `ARC_STARTUP_LANGS` set, generate initial texts
2. Start `background_worker()` as async task
3. Worker runs maintenance cycle every 300 seconds

**Maintenance Cycle** (`services/background_worker.py:maintenance_cycle()`):
1. Detect pool gaps (profiles with < 3 ready texts)
2. Generate texts to fill gaps
3. Retry failed translations
4. Hide old texts (> 30 days, never read)

### Important Patterns

**Profile Selection:**
- Always check `active_profile_id` cookie first
- Fall back to first profile if no active profile
- All pages (reading, words, stats, settings) must respect this cookie

**Timezone Handling:**
- Database datetimes may be naive (no timezone)
- Always check `if datetime.tzinfo is None` and assume UTC if so
- Use `datetime.replace(tzinfo=timezone.utc)` for comparison

**Error Handling in Background Worker:**
- Each gap generation wrapped in try/except
- Failed generations logged but don't crash worker
- `asyncio.gather(*tasks, return_exceptions=True)` to handle partial failures

**LLM Generation:**
- Always check `text.is_ready` before using (content + words_complete + sentences_complete)
- If not ready, show demo text and wait for background worker
- Texts can be in "building" state while translations are being generated

### Testing SRS Changes

After modifying SRS algorithm in `server/services/srs.py`:

1. Drop database: `rm -f ./data/app.db`
2. Restart server: `./run.sh`
3. Create new profile
4. Read texts and click on words
5. Click "Next Text" to save session
6. Check `/words` page to see updated SRS values:
   - **Familiarity**: How well user knows the word (0-1)
   - **Fam Var**: Uncertainty in familiarity (0-0.25)
   - **Decay**: How quickly they forget (0-1 per day)
   - **Decay Var**: Uncertainty in decay rate

### Database Reset Patterns

**Reset users only** (keep global text pool):
```bash
python reset_db.py  # Clears user tables, preserves reading_texts
```

**Full reset** (fresh start):
```bash
rm -f ./data/app.db && ./run.sh  # Deletes DB, restarts server to recreate
```

**Initialize existing data:**
```bash
# After adding new SRS columns, initialize existing lexemes
uv run python server/utils/initialize_srs.py
```

### LLM Model Configuration

Models defined in `server/llm/models.json`:
- Default: `xiaomi/mimo-v2-flash:free` via OpenRouter
- Models assigned to subscription tiers via `allowed_tiers`
- Provider configs in `provider_configs` (timeout, max_retries)

To add new model: Edit `models.json` and restart server.

### Common Issues

**"No available texts" / Demo Text:**
- Background worker hasn't generated texts for your profile yet
- Wait ~5 minutes for next maintenance cycle
- Or set `ARC_STARTUP_LANGS` to include your language and restart

**422 Error `/reading/None/state`:**
- Text_id is "None" (string) instead of actual ID
- Fixed in `reading-state.js` by checking for invalid text_id values

**Timezone error: "can't subtract offset-naive and offset-aware datetimes":**
- Database has naive datetimes (no timezone)
- Fixed in `srs.py:apply_time_decay()` by checking `tzinfo` and assuming UTC

**Words not appearing in /words:**
- Check `active_profile_id` cookie matches expected profile
- Words are per-profile (account_id, profile_id, lang, lemma, pos)
- SRS values initialize after first "Next Text" click

### File Structure Notes

- **`server/models.py`** (3000+ lines): All SQLAlchemy models in one file
- **`server/static/app.js`**: Frontend application state, word click tracking
- **`server/static/reading-state.js`**: Text state manager (fetch, save, clear)
- **`server/templates/pages/reading.html`**: Reading page with interactive word spans
- **`server/templates/pages/words.html`**: Vocabulary list with sorting, filtering
- **`run.sh`**: Changes to project directory, runs `uv sync`, starts uvicorn
