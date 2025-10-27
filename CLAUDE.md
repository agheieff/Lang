# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Arcadia Lang is a click-to-translate web application for language learning, built with FastAPI and featuring pluggable language parsing and dictionary providers. The application supports multiple languages (Spanish, Chinese) with intelligent word analysis, spaced repetition system (SRS), and LLM-powered reading generation.

## Development Commands

### Running the Application
- **Development server**: `uv run uvicorn server.main:app --reload --host 0.0.0.0 --port 8000`
- **Convenience script**: `./run.sh` (handles port configuration and dependency sync)
- **UV script**: `uv run serve` (defined in pyproject.toml)

### Dependency Management
- **Install dependencies**: `uv sync`
- **Add new dependency**: `uv add <package>`
- **Update dependencies**: `uv sync --upgrade`

### Testing
- **Run tests**: `uv run pytest`
- **Single test**: `uv run pytest path/to/test_file.py::test_function`

## Architecture

### Core Components

**FastAPI Application** (`server/main.py`)
- Main application entry point with lifespan management
- CORS middleware for local development
- Rate limiting by IP/user with tier-based limits
- Authentication via JWT tokens and cookie middleware

**Language Processing Pipeline**
- **Tokenization**: Language-specific tokenizers in `langs/tokenize/`
- **Parsing/Analysis**: Language engines in `langs/parsing/` (spaCy for Spanish, jieba for Chinese)
- **Dictionary Providers**: Pluggable providers in `langs/dicts/` (StarDict, CEDICT)

**Spaced Repetition System (SRS)**
- **SRS Logic**: `server/services/srs_logic.py` - Bayesian estimation and scheduling
- **SRS Service**: `server/services/srs_service.py` - API layer for SRS operations
- **Level Tracking**: `server/level.py` - User proficiency level estimation

**LLM Integration**
- **LLM Service**: `server/services/llm_service.py` - Reading generation and translation
- **Translation Service**: `server/services/translation_service.py` - Text translation with caching

### Database Schema
- **SQLite** with SQLAlchemy ORM
- **Models**: `server/models.py` - Profiles, Lexemes, UserLexemes, ReadingTexts, etc.
- **Authentication**: Uses shared `arcadia_auth` library with Account model

### Service Layer Architecture
- **API Routes**: `server/api/` - Profile, wordlists, tiers endpoints
- **Services**: `server/services/` - Business logic separated from web concerns
- **Repositories**: `server/repos/` - Data access layer

## Key Configuration

### Environment Variables
- `ARC_LANG_JWT_SECRET`: JWT signing secret (default: "dev-secret-change")
- `ARC_LANG_MSP_ENABLE`: Enable/disable MStream features
- `ARC_SRS_*`: SRS algorithm parameters (weights, half-lives, etc.)
- `ARC_RATE_*`: Rate limiting configuration per tier
- `ARC_CI_*`: Comprehensible input estimation parameters

### Rate Limiting Tiers
- **Free**: 60 requests/minute
- **Standard**: 300 requests/minute
- **Pro**: 600 requests/minute
- **Pro+**: 1200 requests/minute
- **BYOK**: 100,000 requests/minute
- **admin**: Effectively unlimited

## Language Support

### Spanish (es)
- Tokenization: Client-side regex
- Analysis: spaCy or heuristics for lemma, POS, morphology
- Translation: StarDict/FreeDict dictionaries

### Chinese (zh, zh-Hans, zh-Hant)
- Tokenization: Server-side jieba with CC-CEDICT fallback
- Analysis: Simplified Chinese normalization, pinyin generation
- Translation: CC-CEDICT primary, StarDict secondary

## Development Guidelines

### Code Style
- Use `uv` for all Python operations
- Prefer TypeScript over JavaScript for frontend
- Maximum modularity - easy to swap/adjust components
- Comments only where they explain "why", not "what"
- Clean, production-ready code without legacy cruft

### Authentication Flow
- JWT tokens in Authorization header or cookies
- Tier information embedded in JWT claims
- Account lookup via `get_current_account` dependency
- Tier-based authorization via `require_tier` dependency

### SRS Algorithm
- Bayesian estimation of word familiarity
- Three event types: clicks, non-lookups, exposures
- Exponential decay with configurable half-lives
- FSRS-inspired scheduling with target retention

## File Organization

```
server/
├── main.py              # FastAPI app and routes
├── models.py            # Database models
├── config.py            # Configuration and environment variables
├── deps.py              # FastAPI dependencies
├── level.py             # User level estimation
├── llm.py               # LLM prompt building
├── api/                 # API route modules
├── services/            # Business logic services
├── repos/               # Data access layer
└── templates/           # HTML templates

langs/
├── parsing/             # Language analysis engines
├── dicts/               # Dictionary providers
└── tokenize/            # Tokenization logic
```

## Important Notes

- The application uses a shared `arcadia_auth` library for authentication
- Local development dependencies are managed via editable installs in `pyproject.toml`
- SRS parameters are highly configurable via environment variables
- Chinese language support includes script normalization and pinyin generation
- Rate limiting applies to lookup, parse, and translate endpoints