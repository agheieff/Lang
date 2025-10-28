# Arcadia Lang — Quick Ops Notes

## Parallel word glossing (LLM)

Enable per‑sentence parallel word‑gloss requests to reduce latency while keeping context:

- ARC_WORDS_PARALLEL: integer. If >1, split the generated text into sentences and fire one words request per sentence in parallel. If 1 or unset, use the single‑request path.
- ARC_OR_WORDS_ATTEMPTS: integer. Retry count per sentence for OpenRouter provider only (429/5xx backoff: 2^n with jitter). Other providers use a single attempt.
- ARC_LLM_PROVIDERS: provider order used during reading generation (e.g., "openrouter,local"). The chosen provider/model/base are reused for both structured translations and word glosses.
- LOCAL_LLM_BASE_URL: base URL for the local provider when selected.
- ARC_OR_LOG_DIR: directory for per‑request logs; per‑sentence calls are saved as words_{i}.json; single‑request path uses words.json.
- ARC_OR_LOG_KEEP: how many recent job directories to keep per account/lang (best‑effort retention).

Usage:

```bash
export ARC_WORDS_PARALLEL=6
export ARC_OR_WORDS_ATTEMPTS=2
# optional provider order, default is "openrouter,local"
export ARC_LLM_PROVIDERS=openrouter,local

# run the server
uv run uvicorn server.main:app --reload --host 0.0.0.0 --port 8000
```

Logs for a generation are under:

```
data/llm_stream_logs/<account_id>/<lang>/<timestamp>/
  reading.json
  structured.json
  words.json or words_0.json, words_1.json, ...
  meta.json
```

Notes:

- For Chinese, the words template now uses {sentence}; the code also supports {text} for other templates.
- The server preserves the original reading conversation context for each sentence request to keep glosses coherent.
