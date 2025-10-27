# Parsing and Translation Archive

This directory contains archived parsing and translation functionality that has been removed from the active Arcadia Lang application.

## Archived Components

### NLP Parsing Engine
- `nlp/` - Complete NLP parsing infrastructure
  - `parsing/` - Language-specific parsing engines
    - `es.py` - Spanish parsing with spaCy and fallbacks
    - `zh.py` - Chinese parsing with jieba and OpenCC
    - `registry.py` - Language engine registry
    - `morph_format.py` - Morphological analysis formatting
    - `dicts/` - Dictionary providers
      - `provider.py` - StarDict provider with caching
      - `cedict.py` - CEDICT Chinese-English dictionary
  - `tokenize/` - Tokenization system
    - `base.py` - Tokenizer interface
    - `latin.py` - Latin script tokenizer
    - `zh.py` - Chinese tokenizer with jieba
    - `registry.py` - Tokenizer registry

### API Endpoints
- `parse.py` - Text parsing and tokenization endpoint
- `lookup.py` - Word analysis and dictionary lookup endpoint
- `translation.py` - Translation endpoint with LLM integration

### Services
- `translation_service.py` - Translation service with sentence/paragraph segmentation

## Features Archived

### Language Support
- **Spanish (es)**: spaCy-based parsing, morphological analysis, clitic handling
- **Chinese (zh)**: jieba tokenization, script conversion, pinyin generation

### Dictionary Integration
- **StarDict**: Multi-language dictionary support with compression
- **CEDICT**: Chinese-English dictionary with script conversion

### Tokenization
- Language-specific word segmentation
- Multi-word expression detection
- Character-level analysis for Chinese

### Translation
- LLM-based translation with multiple providers
- Sentence and paragraph segmentation
- Context-aware translation continuation

## Integration Points Removed

The following integration points were removed from the active codebase:
1. Parse, lookup, and translation API routes
2. NLP engine imports in main application
3. Translation service dependencies
4. SRS integration with word analysis

## Usage Notes

This code can be restored if needed, but requires:
- Reintegration of API routes in `server/main.py`
- Reconnection of service dependencies
- Database models for translation logs and lookup results
- Configuration of dictionary file paths

## Dependencies

- spaCy with Spanish models
- jieba for Chinese tokenization
- OpenCC for Chinese script conversion
- StarDict dictionary files
- CEDICT dictionary files