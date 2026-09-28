# Arcadia Lang agent guide

Arcadia Lang is a local, single-user language reader. The app renders prepared lessons and records
learning evidence. The coding agent owns configuration, lesson generation/import, and CSS changes.
Do not add an in-app LLM, API keys, auth, admin pages, or hosted multi-user machinery.

## Toolchain

- Arch Linux; use `pacman`/`yay` for system packages.
- Python dependencies and commands use `uv`; never pip/Poetry.
- Frontend is strict TypeScript/ESM managed with `pnpm`; never npm/yarn.
- FastAPI + SQLAlchemy + SQLite; vanilla TypeScript and CSS.
- Ruff + mypy; Biome + Vitest; 100-column lines; minimal dependencies.
- Keep files small and domain-oriented. Fail fast; implement slightly less error handling than feels
  reasonable.
- `.gitignore` is always a whitelist: block everything, reopen directories, then allow tracked paths.

## Start and verify

```bash
pnpm dev                 # install/sync/build, then bind http://127.0.0.1:8000
./run_tests.sh           # Python + TypeScript checks
uv run lang status       # machine-readable local state summary
```

Runtime state is private/ignored under `data/` (or `$ARC_LANG_DATA_DIR`). `data/lang.db` is the
legacy `es-es` database; new profiles use `data/profiles/PROFILE_ID/lang.db`. Never delete, replace,
or edit databases directly. Interactions are append-only truth; `lexeme_states`, `character_states`,
`grammar_states`, and `proficiency_state` are derived. Lesson skip/restore commands are separate
append-only queue actions and must never be represented as learning interactions.

Activation is explicit and idempotent. Do not forge the reserved `activation` or `onboarding`
preference objects through `profile set`; use the browser form or `profile activate` after the user
chooses to start that language. Inactive profiles must not spend callback tokens, claim generation
jobs, prepare or serve local audio, or invoke browser speech. Manual configuration and imports are
preserved, and activation allows retained lessons to be backfilled later.

## Agent workflow

1. Select or create the language profile, then configure what the user told you. A newly created
   profile is dormant: its existence alone must never trigger lesson or audio generation.

   ```bash
   uv run lang profile list
   uv run lang --profile es-es profile set --learning-language es-ES --translation-language en \
     --level A2 --difficulty 0.27 --interest history \
     --interest "science fiction" --text-length 300
   ```

2. Activate a dormant profile only after the user chooses to start that language. The questionnaire
   seed is optional and never replaces calibration:

   ```bash
   uv run lang --profile PROFILE_ID profile activate --starting-point unsure
   ```

   If the user did not already supply a level, pass their reading-comfort starting point and
   confidence here to seed the first calibration.

3. Read the generation brief: `uv run lang --profile PROFILE_ID brief`.
4. Read the exact lesson schema when needed: `uv run lang lesson schema`.
5. Generate a lesson JSON. Use the brief's due/fragile terms and grammar selectively, mix in
   suitable new material, treat recent openings/endings as repetition warnings, and optimize for
   natural comprehensible input. Choose a specific situation and structure before optional SRS
   terms; lower linguistic difficulty must not collapse every text into the same generic plot.
6. Run `uv run lang lesson validate path.json`, then
   `uv run lang --profile PROFILE_ID lesson import path.json`.
7. Launch or refresh the app for the user.

`pnpm dev` starts provider-neutral automatic generation by default: completion or feedback can
enqueue a callback, and the worker validates and imports its result. Set
`ARC_LANG_AUTO_GENERATE=0 pnpm dev` for an explicitly manual, token-saving run; in that mode,
generate and import lessons in the current agent session.

Validation failures are retained as bounded callback context for the one automatic retry. After the
retry limit, maintenance must leave the failure visible instead of looping; only a changed contract,
profile fingerprint, or a fresh accepted interaction/queue action may create one successor task.
Each attempt refreshes its feedback snapshot, and a worker interruption is recorded as a failed
attempt rather than bypassing the cap. Maintenance uses durable evidence/skip high-water marks to
recover a wakeup missed after an append-only record was committed.

Supported-language lesson imports enqueue cached audio only for an active profile; the durable queue
does not depend on the optional runtime already being installed. The runtime installed by
`scripts/install_qwen_tts.sh` runs in an isolated Python 3.12 worker; never add Torch/Qwen packages
to the main project dependencies. Voice selection is profile-specific, and TTS or preview playback
must never create learning interactions. Keep the existing provider voice catalog and language
notes honest; do not infer regional accents that the fixed voices do not promise.

## Language packs and calibration

Tracked TOML packs in `server/language_rules/` are the source of language-specific generation,
lemma, POS-alias, and learning-unit rules. Base tags apply to their BCP-47 variants; unknown
languages use conservative generic guidance. Add high-confidence declarative rules instead of fuzzy
stemming or ad hoc language branches. Pack changes enter the profile fingerprint and invalidate
stale generation work.

A lesson run is always the display/click boundary, but only a reusable learning unit enters Words,
SRS, calibration, or target selection. The profile-local index aliases only exact normalized
definitions (lemma, normalized POS, gloss, and dictionary pronunciation); it deliberately ignores
frequency rank and never guesses across meanings or homographs. A manual legacy phrase can remain
clickable while resolving to no learning unit. A configured productive span in callback output is
recoverable: the host splits it when exact component definitions are available, or keeps the
display span while projecting learning evidence onto known components. Unknown segmentation and
surface/lemma errors still fail validation. A wrong callback term reference may be rebound only
when the visible surface has one exact known identity; ambiguous or unknown identities remain
errors. On a validation retry, the callback receives its rejected draft and repairs that draft in
place instead of replacing the text.

Chinese dictionary pinyin uses tone marks and one space between syllables. Historical spacing
differences are presentation variants, not separate vocabulary identities; the Chinese pack ignores
whitespace only when comparing dictionary pronunciations. Preserve tones, apostrophes, and other
reading distinctions.

Generated term keys are proposals at the library boundary. When a known key has the same exact
normalized lemma, the host restores its complete stored definition. When that key is already owned
by a different lemma, the host deterministically rekeys the new definition and rewrites its target
and calibration references. Do not try to anticipate the collision suffix; reuse definitions from
the generation brief exactly.

Unknown-level onboarding may store a questionnaire mean/variance, interests, and text length. This
is a prior, never a calibration attempt. Generate typed calibration only when the request contains a
calibration brief. A qualifying attempt needs 30 active seconds, 80% completion, no full
translation, and at least eight usable probes; sentence-translated probes are excluded. Placement
uses timestamped, gradually decaying evidence and must remain replayable. Self-reported levels are
not overwritten by derived placement.

Tracked TOML catalogs in `server/grammar_catalogs/` define stable grammar construction keys,
explanations, categories, rough difficulty, and generation hints. Base catalogs apply to their
BCP-47 variants. Add a catalog entry rather than a shared-code language branch. A lesson occurrence
references one catalog key and anchors it to a precise half-open sentence run range.

## Lesson and evidence rules

- Lesson text is `blocks -> sentences -> runs`; a run is plain text or a term-bearing text fragment.
  Never use HTML or character offsets. The UI creates text nodes, so Unicode and repeated terms remain
  safe.
- A term key is stable for one `(language, lemma, POS)` meaning. Reuse its full definition exactly.
- Use BCP-47-style variants: `es-ES`, `es-419`, `zh-Hans`, `zh-Hant`. Keep CEFR `level`
  broad and use `difficulty` (0-1) for finer adaptation.
- Agent-generated drafts must annotate every lexical token, not only SRS targets or calibration
  probes. Plain runs contain only whitespace/punctuation; give every generated term a consistent
  approximate corpus `frequency_rank` (1 is most common). Follow the request's `language_guidance`;
  segment Chinese into reusable dictionary words and adjacent grammatical units, not productive
  number/determiner-plus-classifier phrases.
- Keep display/click spans separate from canonical learning units. Add language-specific POS aliases,
  lemma guidance, productive-expression exclusions, and narrow exceptions in
  `server/language_rules/LANGUAGE.toml`; avoid hard-coding a new language in the shared reducers.
- Keep dictionary pronunciation on the canonical term. Use optional `run.pronunciation` only for a
  contextual surface reading such as tone sandhi; it overrides display for that occurrence without
  creating a new SRS identity.
- Prefer 0-3 deliberate SRS targets per lesson. Record them in `target_term_keys`.
- For ordinary callback generation, follow the request's `content_plan` without quoting it. Vary
  discourse form, perspective, immediate goal, progression, and ending independently of topic. Do
  not default to a group gathering, a token problem, quick collective success, and an explicit
  everyone-is-happy moral. Select SRS candidates after the content shape and use a lower-ranked
  candidate or none rather than distorting natural text. Persisted plans and capped recent excerpts
  are generation context; technical target/grammar bookkeeping is not.
- Treat `priority_grammar` as a ranked choice pool, never a checklist. Natural text quality wins and
  zero deliberate uses is valid. Annotate every clearly present catalog construction, including
  incidental ones, in `sentence.grammar`; use lesson-unique occurrence keys, precise half-open run
  ranges, and optional notes only for context-specific help.
- Only use typed `calibration` probes when the generation request asks for them. Use every requested
  difficulty exactly once, keep probe terms unique and previously unused, and make each occur once.
- Words and characters use the FSRS-4.5 memory model (`server/memory_model.py`). A term reveal is
  an explicit lapse ("again"), capped at one per term per session. A sufficiently read, completed
  lesson gives each unrevealed term a partial "good" review weighted by `passive_confidence`;
  sentence/full translations suppress that passive evidence for their scope. A reread of an
  already-read lesson is discounted by `reread_weight` and is not a new qualified exposure. Massed
  exposure is nearly worthless by construction: FSRS barely raises stability while recall is
  already high. A reveal of a configured productive span distributes one bounded failure unit over
  its selected components, weighted toward the less-known component; it never gives every homograph
  the full penalty.
- `mastery` is time-aware: the probability of still recalling a unit one horizon (30 days) from
  now. Words without evidence use a frequency prior around the learner's frontier rank, and the
  prior also seeds the first review so pre-existing knowledge of common words survives. Keep
  `alpha`/`beta` as evidence tallies for uncertainty only. Do not replace any of this with raw click
  or appearance counts, and re-score parameter changes with `lang memory evaluate --fit` against
  real history before adopting them.
- Chinese character recognition is another replayable projection of those same events, never a new
  interaction stream. A qualified clean session supplies one bounded positive observation per
  character, plus a small bonus for a new canonical word context; repeated appearances inside that
  session are not independent successes. A one-character reveal is bounded negative evidence, while
  a longer display span shares a smaller total across its distinct characters, weighted toward
  weaker/less-certain estimates. Deduplicate repeats and cap each character per lesson session.
  Sentence/full help suppresses passive credit in scope. Keep the character-specific memory policy
  independently configurable from words.
- Opening a grammar marker is neutral. Revealing its explanation records explicit negative evidence
  through `translation.revealed` with `scope="grammar"`; repeat penalties are capped per
  construction/session. A sentence translation can be weaker inferred grammar evidence when
  vocabulary reveals do not explain it. Qualified unassisted completion is weak positive evidence;
  sentence/full help suppresses it for the corresponding scope. Count at most one clean success per
  construction/session, with only a small first-clean-reading bonus for a new lesson context.
  Grammar mastery and stability weights remain independently configurable from words and characters.
- Reader marker suppression is only a projection: keep every authored grammar occurrence and all
  reducer evidence. Hide a construction only after conservative, varied, stable evidence while its
  due date remains in the future; overdue, fragile, unknown, and missing-due constructions stay
  visible. In mixed sentences, suppress individual comfortable occurrences rather than the whole
  marker.
- Skipping is a reversible queue disposition, not completion, rating, or learning evidence. Exclude
  actively skipped lessons from the reader's default queue and replacement-queue count, but keep
  them explicitly accessible for peeking and restoration. Preserve any genuine interactions that
  occurred before the skip, and let the latest append-only skip/restore action determine
  disposition.
- Completion never requires a rating. Record `lesson.completed` whenever the user finishes; add
  `lesson.rated` only when a rating or optional feedback tag is present. A null rating is valid only
  when the event contains feedback.
- Treat `priority_terms` and `target_policy.candidate_term_keys` as ranked opportunities, never a
  checklist. Urgency ranks reviewed words by 1 - predicted recall now (a bonus once recall falls
  below 90%) and never-reviewed words by how likely they are still unknown, plus frequency,
  uncertainty, and repeated failures. Natural text quality may legitimately result in zero
  realized targets.
- New vocabulary comes from two sources on purpose: `brief.vocabulary.list_candidates`
  (frequency-ordered word lists in `server/word_lists/`, restricted to unencountered words near the
  learner's level frontier) and a few words the agent chooses itself. The agent's own picks are a
  deliberate random element and the only source once a list is exhausted or a language has none.
- Every import records `known_share_at_import`, the predicted share of running words the learner
  knows. The next plan compares recent values with the target (95% by default,
  `target_known_share` preference) and scales new-word counts; texts show their current share.
- For Chinese candidate ordering only, character retrievability may provide the existing small,
  bounded readability hint for sparse-evidence words. It fades out after direct word evidence and
  must never alter lexeme mastery, evidence, bands, urgency values, or due dates.
- Do not reward a mere render/scroll as knowledge. Do not mutate derived SRS values manually.
- Statistics are read-only projections. Count vocabulary through the same canonical opened-text
  boundary as Words; count read texts by distinct completed lesson and rereads by completed session.
  Total tracked time comes only from completed-session active timers. Representative WPM uses the
  token/time-weighted latest ten completions meeting the existing 30-second/80-percent evidence
  threshold; unfinished browser-local time is unavailable to the server.
- The browser's current active-time policy is one configurable five-second lease after pointer
  movement. Start, resume, reset, and visibility changes stay disarmed until fresh movement; do not
  silently broaden those into activity signals.
