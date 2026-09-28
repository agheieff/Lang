# Word lists

Frequency-ordered vocabulary lists that seed new-word offers for a language. Generation also asks
the agent for a few new words of its own choice, so new vocabulary keeps arriving after a list is
exhausted.

| File | Languages | Source | License |
| --- | --- | --- | --- |
| `zh-hsk.tsv` | `zh`, `zh-Hans`, `zh-Hant` | HSK 3.0 levels 1-7 (7 = 7-9) with HSK 2.0 fallback, from [drkameleon/complete-hsk-vocabulary](https://github.com/drkameleon/complete-hsk-vocabulary) | MIT, see `LICENSE-complete-hsk-vocabulary.txt` |

Columns: `lemma`, `pronunciation`, `gloss`, `frequency_rank`, `level`. Rebuild with
`uv run python scripts/build_zh_word_list.py complete.min.json`.
