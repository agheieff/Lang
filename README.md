# Arcadia Lang

A click-to-translate web application for language learning, featuring pluggable language parsing and dictionary providers with intelligent spaced repetition.

## Overview

Arcadia Lang generates reading texts tailored to individual learner levels, providing:

- Custom text generation based on user proficiency and SRS data
- Sentence-by-sentence translations
- Detailed word analysis with linguistic annotations
- Support for Spanish and Chinese languages

## Text/Translation Generation Pipeline

The system uses a multi-stage pipeline:

1. **Text Generation**: Creates targeted reading content based on user level and vocabulary needs
2. **Structured Translation**: Provides sentence-by-sentence translations
3. **Word Analysis**: Detailed linguistic analysis with translations, lemmas, and grammar
4. **Retry System**: Automatic recovery from failed components

## Configuration

### Parallel Word Glossing

Enable per‑sentence parallel word‑gloss requests to reduce latency while keeping context:

- `ARC_WORDS_PARALLEL`: integer. If >1, split the generated text into sentences and fire one words request per sentence in parallel. If 1 or unset, use the single‑request path.
- `ARC_OR_WORDS_ATTEMPTS`: integer. Retry count per sentence for OpenRouter provider only (429/5xx backoff: 2^n with jitter). Other providers use a single attempt.
- `ARC_LLM_PROVIDERS`: provider order used during reading generation (e.g., "openrouter,local"). The chosen provider/model/base are reused for both structured translations and word glosses.
- `LOCAL_LLM_BASE_URL`: base URL for the local provider when selected.
- `ARC_OR_LOG_DIR`: directory for per‑request logs; per‑sentence calls are saved as words_{i}.json; single‑request path uses words.json.
- `ARC_OR_LOG_KEEP`: how many recent job directories to keep per account/lang (best‑effort retention).

## Running the Application

```bash
# Ensure dependencies are installed
uv sync

# Set environment variables for parallel processing
export ARC_WORDS_PARALLEL=6
export ARC_OR_WORDS_ATTEMPTS=2
# optional provider order, default is "openrouter,local"
export ARC_LLM_PROVIDERS=openrouter,local

# Start the development server
uv run uvicorn server.main:app --reload --host 0.0.0.0 --port 8000

# Or use the convenience script
./run.sh
```

## Architecture

### Core Components
- **FastAPI Application** (`server/main.py`) - Main web server with API routes
- **Generation Queue** (`server/services/gen_queue.py`) - Async text generation pipeline
- **LLM Integration** (`server/llm/`) - Language model communication and prompt management
- **Database Models** (`server/models.py`) - SQLAlchemy ORM for text storage

### Service Layer
- `generation_orchestrator.py` - High-level coordination
- `llm_common.py` - Prompt building and user context
- `retry_service.py` - Automatic error recovery
- `readiness_service.py` - Component availability checking

## Logging & Monitoring

Generation logs are stored under:

```
data/llm_stream_logs/<account_id>/<lang>/<timestamp>/
  text.json              # Main text generation request/response
  structured.json        # Sentence translations
  words_0.json, ...      # Per-sentence word translations
  meta.json              # Job metadata and status
```

### Language Support
- **Spanish**: Uses regex tokenization, spaCy/StarDict integration
- **Chinese**: Jieba tokenization with pinyin generation, CC-CEDICT integration

## Development

### Installing Dependencies
```bash
uv sync                    # Install all dependencies
uv add <package>          # Add new dependency
```

### Testing
```bash
uv run pytest             # Run all tests
uv run pytest path/to/test.py::test_function  # Run specific test
```

### Code Style
- Use `uv` for all Python operations
- Modular architecture with easy component swapping
- Comments explain "why" not "what"
- Clean production-ready code

## Notes

- For Chinese, word translation templates use `{sentence}`; other languages support `{text}` placeholders
- The server preserves original reading conversation context for each sentence to maintain translation coherence
- Parallel word processing significantly reduces latency for longer texts
- File-based cross-process locking prevents duplicate generations
