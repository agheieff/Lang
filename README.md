# Arcadia Lang

A language learning application with LLM-generated reading practice, click-to-translate word glosses, and sentence translations.

## Overview

Arcadia Lang generates personalized reading texts for language learners:

- **Global text pool** - Texts are generated once and shared across users with matching language pairs
- **Multi-profile support** - Users can learn multiple languages with separate profiles
- **Click-to-translate** - Word glosses and sentence translations appear on hover/click
- **Background generation** - Texts are pre-generated so they're ready when users need them

## Quick Start

```bash
# Install dependencies
uv sync

# Set required environment variables
export ARC_LANG_JWT_SECRET="your-secret-key"
export OPENROUTER_API_KEY="your-openrouter-key"  # or use local LLM

# Start the server
./run.sh
# or: uv run uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000`, create an account, set up a profile with your target language, and start reading.

## Architecture

### Database Structure

**Global database** (`data/arcadia_lang.db`):
- `accounts` - User accounts with auth
- `profiles` - Language learning profiles (one per language pair per user)
- `reading_texts` - Generated texts shared across users
- `reading_text_translations` - Sentence translations
- `reading_word_glosses` - Word-level translations

**Per-account databases** (`data/user_<id>.db`):
- `lexemes` - User's vocabulary knowledge
- `word_events` - Interaction history
- `user_reading_history` - Which texts the user has read

### Key Components

- **Background Worker** (`server/services/background_worker.py`) - Manages text pool, retries failed translations
- **Generation Orchestrator** (`server/services/generation_orchestrator.py`) - Coordinates text + translation generation
- **LLM Providers** (`server/llm/`) - OpenRouter, local LLM, and provider fallback chain

### Text Generation Pipeline

1. User requests reading → system checks for available texts in pool
2. If pool is low, background worker triggers generation
3. Text content generated via LLM
4. Parallel requests for word glosses and sentence translations
5. Text marked "ready" when all components complete
6. Failed translations automatically retried by background worker

## Configuration

### Required
- `ARC_LANG_JWT_SECRET` - JWT signing secret
- `OPENROUTER_API_KEY` - For OpenRouter provider (or configure local LLM)

### Optional
- `ARC_LLM_PROVIDERS` - Provider order, e.g., `"openrouter,local"` (default: `"openrouter,local"`)
- `LOCAL_LLM_BASE_URL` - Base URL for local LLM provider
- `ARC_WORDS_PARALLEL` - Number of parallel word gloss requests (default: 1)
- `ARC_OR_WORDS_ATTEMPTS` - Retry count for OpenRouter word requests

### Startup Pre-generation
- `ARC_STARTUP_LANGS` - Comma-separated languages to pre-generate on startup (e.g., `"es,zh"`)
- `ARC_STARTUP_TEXTS_PER_LANG` - Texts to generate per language on startup (default: 2)

Example:
```bash
export ARC_STARTUP_LANGS="es,zh"
export ARC_STARTUP_TEXTS_PER_LANG="3"
```

## Development

```bash
# Run tests
./run_tests.sh
# or: uv run pytest tests/

# Run specific test
uv run pytest tests/unit/path/to/test.py -v
```

## Admin Pages

Admin users can access monitoring pages:
- `/admin/texts` - View all texts in the pool with status (ready/pending/failed)
- `/admin/accounts` - View accounts, profiles, change subscription tiers

## Language Support

Currently supports:
- **Chinese** (zh) - Jieba tokenization, pinyin generation
- **Spanish** (es) - Regex tokenization

Target language for translations is typically English (en).
