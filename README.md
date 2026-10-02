# Arcadia Lang

A local language-learning reader operated by Codex or another coding agent.

The browser app has one job: present prepared reading lessons, reveal translations, and record what
the learner needed help with. The agent handles profile changes, lesson generation, content imports,
and visual updates. There is no embedded model, account system, cloud service, or API key.

## Run locally

On Arch Linux:

```bash
sudo pacman -S --needed uv nodejs pnpm
git clone git@github.com:agheieff/Lang.git
cd Lang
pnpm dev
```

Open <http://127.0.0.1:8000>. `pnpm dev` syncs locked dependencies, builds the TypeScript UI, creates
the local SQLite database, and starts FastAPI. After dependency installation, the app itself does not
need network access. Browser-provided speech voices are a separate operating-system/browser service
and are not guaranteed to be offline.

## First lesson

New workspaces are dormant until the learner explicitly starts that language. The browser's short
onboarding form can record a rough reading starting point, confidence, interests, and preferred
text length; an agent can perform the same explicit action with `profile activate` after the
user chooses to begin. Activation is what permits automatic generation and listening work. Merely
creating or configuring a profile does not spend callback tokens, enqueue neural audio, or invoke
browser speech.

Automatic generation is enabled by default, so activation can enqueue the first calibration or
ordinary text. Start with `ARC_LANG_AUTO_GENERATE=0 pnpm dev` when the coding agent should prepare
and import lessons manually instead.

An agent can set up a learner and import a generated lesson with:

```bash
uv run lang profile set \
  --learning-language es-ES \
  --translation-language en \
  --level A2 \
  --difficulty 0.27 \
  --interest history \
  --interest cooking \
  --text-length 300

uv run lang profile activate --starting-point unsure

uv run lang brief
uv run lang lesson schema
uv run lang lesson validate examples/es-demo.json
uv run lang lesson import examples/es-demo.json
```

The CLI prints JSON so any coding agent can inspect and update the app without scraping the UI.

## Language profiles

Each language track has its own SQLite database and private agent workspace. The original Spanish
state remains in `data/lang.db` and is registered as `es-es` without being copied or moved. New
profiles live under `data/profiles/PROFILE_ID/`.

```bash
uv run lang profile list
uv run lang profile create zh-hant \
  --label "Chinese (Traditional)" \
  --learning-language zh-Hant \
  --translation-language en \
  --select
uv run lang --profile es-es brief
uv run lang --profile zh-hant profile set --level A1 --difficulty 0.1

uv run lang profile create de-de \
  --label "German" \
  --learning-language de-DE \
  --translation-language en
```

`profile select` sets the CLI default. `--profile` takes precedence, followed by
`ARC_LANG_PROFILE`. The browser profile switch has its own URL, event queue, and lesson sessions, so
multiple profile tabs cannot mix progress.

`profile delete PROFILE_ID` removes one non-selected, isolated profile workspace in full, including
its database, generated artifacts, and audio. It refuses the selected profile and the shared legacy
Spanish layout.

`profile reset PROFILE_ID` clears one non-selected profile's lessons, evidence, generated artifacts,
and audio while preserving its language entry and SQLite file. The recreated profile is inactive
and ready for the starting questionnaire; this also supports the legacy Spanish layout. Stop the
development app first so generation and audio workers cannot race the reset.

The picker presents human language/variant names rather than raw tags. It uses region flags where a
real region is present, a Latin America marker for `es-419`, and script-aware Chinese markers.
`zh-Hans` uses China only as an explicit simplified-script picker marker; unqualified `zh-Hant`
stays region-neutral instead of implying Taiwan. Accessible option labels spell out the marker's
meaning.

### Activation and placement

New profiles remain inactive even if the agent configures their interests or level first. Until an
explicit browser or CLI activation, the app shows the start screen and automatic queue maintenance,
topic requests, generation-task claims, local audio preparation, cached playback, and browser-speech
fallback all remain dormant. Manual configuration and lesson imports are preserved, but inactive
imports do not enqueue audio. Activation is idempotent; existing installations are migrated as
active only when their workspace already contains lessons.

If the level is unknown, the stored near-B1 value is only a neutral internal seed. The optional
questionnaire replaces it with a broad prior based on reading comfort; the confidence answer changes
the prior's width, not its status as evidence. “Not sure” supplies no prior. Interests and requested
text length shape generation but do not count as proficiency evidence.

Automatic generation requests one calibration text at a time. A calibration draft has 12 unique,
previously unused probe terms spread over requested difficulties, while all of its ordinary
vocabulary remains annotated and clickable. An attempt qualifies only after at least 30 active
seconds, 80% completion, no full translation, and at least eight probes outside
sentence-translated scopes. The questionnaire is never counted as an attempt. Two qualified texts
normally produce a rough estimate; four or more can become stable once the uncertainty interval is
narrow enough. Older timestamped evidence fades gradually, so placement can adapt after a long gap.

Supplying `--level` or `--difficulty` marks the profile as self-reported; activation with
`--starting-point unsure` preserves it, and derived placement does not overwrite it. Choosing a
specific questionnaire starting point instead deliberately replaces that value with a revisable
unknown-level prior. After placement, the first qualified completion of each ordinary lesson adds
low-weight level evidence. Word reveals are deliberately weak because they may be pronunciation or
meaning checks; sentence/full translation and explicit easier/more-challenging feedback carry more
weight. Rereads do not move overall level. A manual level change creates a stable evidence anchor,
so older readings cannot immediately undo the correction.

## Learning loop

1. The learner reads a generated text.
2. Clicking a term reveals its gloss and records an uncertainty signal. One isolated check is weak
   evidence; reveals in later sessions progressively strengthen it. If a legacy click span contains
   known productive components, one bounded signal is divided between them, with more assigned to
   the component whose current mastery is lower.
3. Sentence, grammar, or full-text help is available in layers. Opening a grammar marker is neutral;
   revealing its explanation records one direct-help signal for that occurrence and session. With a
   mouse, clicking between words opens a sentence's translation. On touch, a stray tap does not:
   long-press the sentence or use the floating Sentence button, and a translation closed within
   1.5 seconds is treated as a mistap and not recorded.
4. Completing a meaningfully read lesson gives weak recognition evidence only to terms that were not
   revealed in a translated scope.
5. The app rebuilds deterministic per-term, per-character, and per-construction learning state from
   raw history.
6. Authored calibration probes establish placement. Qualified first readings then refine it with
   lower-weight evidence; rereads, short/incomplete sessions, and uncalibrated unknown profiles do
   not move it accidentally.
7. `uv run lang brief` ranks due, fragile, and unseen terms and grammar. The agent chooses a small,
   natural subset for the next text, balanced with level, interests, and recent content.

This keeps the distinctive part of the original prototype—implicit testing through reading—without
making every unclicked render count as knowledge.

Vocabulary state is replayed chronologically from append-only interactions through the FSRS-4.5
memory model (`server/memory_model.py`). A reveal is a lapse ("again"); a clean, qualified reading is
a partial success ("good", weighted by `passive_confidence` because not clicking is weaker evidence
than a correct answer). Rereading a text is discounted further and is not a new exposure. FSRS barely
raises stability while predicted recall is still high, so massed exposure adds almost nothing, and
lapses shorten stability according to how surprising they were. Multiple raw reveals remain
inspectable, but only the first reveal of a term in one lesson session affects the rebuilt state.

Mastery is the time-aware probability of still recalling a word 30 days from now; it decays without
evidence. A word without evidence uses a frequency prior around the learner's frontier rank, and that
prior also seeds the first review, so common words a learner already knows are not treated as new.
`uv run lang memory evaluate --fit` scores the model one step ahead against real reveal history (for
the zh-hans history: log-loss 0.43 and AUC 0.84, against 0.54 and 0.73 for the earlier Beta model).

Review priority ranks reviewed words by 1 - predicted recall now, with a bonus once recall falls below
90%, and never-reviewed words by how likely they are still unknown, plus corpus frequency,
uncertainty, and repeated failures across lessons. A rare one-off context lookup is down-weighted.
New vocabulary comes from frequency word lists (`server/word_lists/`, currently HSK for Chinese)
restricted to unencountered words near the learner's level, plus a few words the agent chooses itself
so new words keep arriving after a list is exhausted. Each import records its predicted share of known
running words; new-word counts steer toward a 95% target and text cards show each text's share.
Terms offered but not naturally used are cooled for the next offer instead of being forced into the
text.

Subjects follow **reading preferences**: free-text notes in Settings that start from your listed
interests, which you can edit and which the agent maintains. Messages sent from Settings, ratings,
skips, and abandoned texts are the signals; after a message or every few reactions the worker asks
the agent for an updated revision (history is kept). Each new text is a favourite (60%), a variation
of an interest (25%), or a new subject (15%); variation and new-subject texts state the question they
test, shown when you rate the text. The agent sees the last 20 texts with their reactions, so it
avoids repeats and leans toward what worked.
For Chinese, inferred character retrievability can move an otherwise-unseen word's candidate order
by at most 15%, making a word built from readable characters slightly easier to introduce. This
hint fades to zero after three direct word-level signals and never changes that word's mastery,
evidence, knowledge band, or due date.

Term underlines expose the current estimate without changing it: familiar (hidden until hover),
expected (quiet green), uncertain (amber), deliberate learning focus (blue), and incidental context
above the current level (muted violet). Only reading events change learning state; the bands are
recomputed from that state for each lesson.

The Words rail shows canonical terms from lessons that were actually opened; prepared-but-unread
lessons stay out of the list. A row is one stable lemma, part of speech, and meaning, with observed
surface forms grouped underneath. Other observed meanings or grammatical roles are linked while
keeping their evidence separate. Every column is sortable, and search covers word classes plus
tone-insensitive, spacing-insensitive pinyin. The list separates raw popup clicks from
lesson/session-deduplicated reveals that affect SRS, and also shows occurrences, clean passive
reads, mastery, and due state. Opening the list is read-only. The small reset control beside active
reading time only resets the local clock for the current lesson session and does not record learning
evidence. The motion policy is deliberately small: mouse movement opens a five-second countable
window, and on touch devices a tap, touch movement, or scroll opens a fifteen-second window, since
phone reading has no continuous pointer movement. Counting then pauses until the next activity.
Starting, resuming, resetting, or returning to the tab does not open a window by itself.

Chinese profiles also expose a **Characters** rail. Every literal Han character in the title and
body of an opened text is counted, including characters in display-only or older plain runs;
simplified and traditional characters remain distinct. Alongside those observations, a replayable
derived state estimates character recognition from the same append-only reading history. Each clean,
qualified reading session supplies one bounded positive observation per character, plus a small
bonus when it is met in a new canonical word context. Repeated appearances inside one session do not
multiply into independent successes. A reveal of a one-character run supplies bounded negative
evidence; a multi-character reveal shares a smaller total across its characters, weighted toward
the currently weaker or less certain character. Repeats are deduplicated and each character is
capped per lesson session. Sentence and full-text help suppress passive credit in their scopes.

Characters use the same FSRS model with their own policy: inferred reveals are partial lapses sized by
the failure mass, and clean sessions are partial successes with a new-context bonus. Retrievability is
current predicted recall and mastery is recall 30 days ahead. The initial prior is deliberately
neutral, and a character with no qualifying evidence is displayed as unestimated rather than known. Word evidence therefore helps estimate its component characters;
character knowledge only provides the small, fading generation-order hint described above. It does
not leak back into word SRS. Direct character-test counters are reserved for a future dedicated
exercise and remain zero for ordinary reading.

The Statistics rail is another read-only projection. **Words met** uses the same canonical,
opened-text vocabulary boundary as Words and groups the 0-1 mastery estimate into low-confidence,
developing, familiar, and strong ranges. **Texts read** counts unique completed lessons, while
completed sessions also expose rereads. Tracked active time sums the active timers sent with
completed sessions; unfinished session time remains browser-local. Recent pace is a token/time
weighted WPM over at most ten latest completions that reached 30 active seconds and 80% completion,
which excludes accidental test finishes. Chinese pace counts authored segmented word units rather
than individual characters. The level card shows the same current adaptive estimate used by the
reader and lesson generator, together with its calibration range. A self-report or questionnaire
answer remains the replay prior rather than a competing current value.

The Grammar rail follows the same opened-text boundary. Its cards group authored construction
occurrences, explanations, contextual examples, direct explanation reveals, weak sentence-help
inferences, clean completed readings, and a conservative confidence/due estimate. A single check is
not labelled a problem, repeated raw taps remain inspectable without being double-counted, and the
UI never claims a construction is permanently mastered. One qualified clean construction/session
adds one bounded positive observation; its first clean reading in a new lesson receives a small
context bonus, while repeated occurrences inside that lesson do not multiply into independent
trials. Grammar stability has its own configurable days-per-evidence coefficient and delayed-recall
bonus. In the reader, a quiet `G` marker first shows which construction is present without recording
evidence; only **Show explanation** records help. After strong, varied, stable evidence, a
not-yet-due construction loses its marker just as a familiar word loses its underline. It returns
when review is due or later evidence weakens the estimate; mixed sentences keep only the still-useful
notes. `Peek` mode records nothing.

The Texts rail is the reading library. It separates ready, in-progress, skipped, and previously read
texts. Pending and running generation jobs appear separately under **Being prepared** without
inventing a title, difficulty, or word count before the lesson exists; the library polls until each
placeholder becomes a real ready text. **Skip for now** removes an unfinished text from the active
queue without completing it or creating learning evidence; any real evidence already recorded
before the skip remains intact. The text stays recoverable under **Skipped**, where it can be
returned to the queue, opened immediately, or peeked at. Because skipped texts no longer count
toward the usable queue, automatic generation may prepare a replacement. Ready texts have compact
up/down controls; each move is an append-only adjacent swap, and the resulting durable order also
chooses the next default Reading text. `Peek` opens the complete reader without creating a session,
timer, Words exposure, or learning event. `Read again`
deliberately starts a fresh evidence-bearing session. A topic request creates a durable generation
job even when the normal unread queue is already full.
Here, the **title** is the reader-facing target-language heading; the **topic** is a short
organizational subject used for requests, filtering, continuity, and generation diversity. Lesson
rating is optional: **Finish & next** always records completion, while rating or fine-tuning
feedback is recorded only when supplied.

### Language packs and learning units

Runs are the immutable display and click boundary; learning units are the narrower vocabulary/SRS
boundary. Tracked TOML language packs under `server/language_rules/` provide generation guidance,
part-of-speech aliases, lemma conventions, and conservative include/exclude rules. `de`, `es`, and
`zh` currently have specialized packs, their regional/script variants inherit the base rules, and
other tags use safe generic guidance. Pack contents are included in generation fingerprints, so a
rule change invalidates stale generation work.

The learning-unit index never fuzzy-stems or merges by spelling alone. Exact matches of normalized
lemma, normalized part of speech, gloss, and dictionary pronunciation share the first canonical SRS
identity; frequency rank is deliberately not part of that identity. Different meanings,
pronunciations, or grammatical roles remain separate. Existing/manual lessons can retain a
clickable phrase annotation even when it is not a reusable unit. The phrase itself remains
context-only; configured reusable components may appear in Words and receive projected evidence.

Generated lessons are asked for adjacent reusable runs. When Mandarin output still contains a
configured productive combination, the host splits it if every exact component definition is
available; otherwise the natural display span can remain while the learning layer projects it onto
known components. This prevents a known segmentation choice from discarding a complete text.
An unrelated callback key is rebound only when the visible surface has one exact local or stored
identity. Unknown or ambiguous identities still fail; the retry then receives the rejected compact
draft and repairs its annotations in place rather than generating a replacement story. Spanish and
German guidance keeps dictionary lemmas without guessing through inflection or merging homographs.

Tracked grammar catalogs under `server/grammar_catalogs/` define stable construction keys, learner
explanations, categories, rough 0-1 difficulty, and generation hints. Base catalogs currently cover
Spanish, German, and Mandarin and apply to their regional/script variants. Lessons anchor a catalog
construction to a sentence with a lesson-unique occurrence key and a half-open run range; an
optional note explains only what is special about that use. Old lessons can omit grammar entirely.
Catalog contents enter generation fingerprints.

Mandarin pronunciation is displayed as space-separated pinyin syllables when that alignment can be
established. Erhua keeps its final `r` on the preceding syllable and extends that syllable's color
over a written `儿`; otherwise the usual one-syllable-per-Han alignment applies. Joined and spaced
historical spellings resolve to the same Chinese vocabulary identity when the lemma, part of speech,
meaning, and actual reading match; new generation uses one space between syllables. Tone colors
follow Pleco's order: red, green, blue, purple, and gray for tones one through four and the neutral
tone. Chinese profiles also expose a profile-scoped Hanzi-tone toggle; it colors source characters
only when the pronunciation alignment is unambiguous. The canonical term stores its dictionary
pronunciation. An optional
`run.pronunciation` overrides it only for that occurrence, allowing contextual readings and tone
sandhi without splitting one lemma into false vocabulary identities; the popup and Hanzi/pinyin
rendering prefer the override.

For presentation, Chinese lesson difficulty is shown as an explicitly approximate HSK 1-9 band;
the stored CEFR-compatible level and continuous 0-1 adaptation model remain unchanged. The reader
places level, fine-grained difficulty, and the labelled topic after the text rather than crowding the
title. Settings provides a browser-local light/dark preference shared across language profiles and a
separate reading voice saved for each language profile.

## Optional local speech

After a language is activated, Listen works immediately with a compatible browser or
operating-system voice. For cached local neural audio, install the isolated Qwen3-TTS runtime on
this AMD ROCm host:

```bash
./scripts/install_qwen_tts.sh
pnpm dev
```

The installer uses `uv`, Python 3.12, AMD's pinned ROCm wheels, and a pinned revision of
`Qwen3-TTS-12Hz-0.6B-CustomVoice`. The heavyweight Torch environment stays under ignored
`data/tts/`; the main app never imports it. `pnpm dev` starts one audio worker when that runtime is
ready. Every supported-language lesson imported into an active profile enqueues an immutable,
profile-local WAV without blocking the import; the queue can exist before the optional runtime is
installed. Changing the voice in an active profile supersedes obsolete pending work and backfills
all texts one at a time. A voice selected while inactive is saved without starting work. Activation
allows the worker or audio-status path to backfill existing lessons, while cached files and imported
content are retained throughout. Old audio never masquerades as the newly selected voice.

Qwen's compact model supports Chinese, English, Japanese, Korean, German, French, Russian,
Portuguese, Spanish, and Italian with nine fixed voices. It reads both `zh-Hans` and `zh-Hant` as
Mandarin while preserving the source script; it does not imply Cantonese or a Taiwan voice. It reads
both `es-ES` and `es-419` as Spanish, but its presets do not guarantee a Spain or Latin American
accent. German uses Qwen's German mode and starts with Ryan; because none of the nine presets is a
native-German voice, the voice selector does not promise a native regional accent. The 0.6B model
has no dependable prompt-based accent, emotion, or speed control, so playback speed is handled by
the reader. Voice cloning and arbitrary synthesis endpoints are intentionally out of scope.

This machine uses eager FP32 generation because its Arch/RDNA4 ROCm stack crashes in the faster
FP16 attention path. Other installations may explicitly set `ARC_LANG_TTS_ATTENTION=sdpa` and
`ARC_LANG_TTS_DTYPE=float16` after a synthesis smoke test. Set `ARC_LANG_TTS_ENABLED=0` to leave the
local worker off; browser speech remains available for activated profiles.

## Neural audio on another machine

The reader can run on a small always-on host while speech is synthesized on a stronger machine.
Set `ARC_LANG_TTS_REMOTE=1` for the host's web service so audio status reports "queued" instead of
"unavailable". On the machine with the Qwen runtime (`scripts/install_qwen_tts.sh`), run
`uv run python -m server.remote_tts_worker --host HOST` or install `scripts/lang-tts-remote.service`.
It claims one task at a time with `lang tts claim` over SSH, synthesizes locally, and uploads the WAV
with `lang tts complete`. Each claim returns a `claim_token`; complete, fail and release require
`--claim TOKEN`. Reassigned claims reject an old worker's results. Remote queue operations are
serialized on the owner, and an exact repeated upload returns the existing completion. A claim left
unfinished for 45 minutes returns to the queue without using an attempt; synthesis times out after
40 minutes so reporting fits inside that lease. Network loss leaves uncertain work for owner
readback/reclaim, while reported synthesis failures retain the two-attempt cap. WAV validation checks
all declared frames before publication. Nothing is exposed beyond SSH. Run either the local worker
or the remote queue on a host, never both. Upgrade the owner and PC worker together: stop the idle PC
worker, deploy the owner, then restart the PC worker.

## Commands

```bash
uv run lang profile show
uv run lang profile list
uv run lang profile create --help
uv run lang profile delete PROFILE_ID
uv run lang profile reset PROFILE_ID
uv run lang profile select PROFILE_ID
uv run lang profile set --help
uv run lang profile activate --help
uv run lang brief
uv run lang lesson schema
uv run lang lesson validate PATH
uv run lang lesson import PATH
uv run lang lesson list
uv run lang generation ensure
uv run lang generation list
uv run lang generation retry TASK_ID
uv run lang rebuild
uv run lang status
./run_tests.sh
```

The profile registry and runtime data live under `data/` and are ignored by Git. Set
`ARC_LANG_DATA_DIR` to use another root. The legacy `data/lang.db` remains the `es-es` database;
new profile databases and agent jobs are isolated below `data/profiles/PROFILE_ID/`.

Languages use BCP-47-style tags, so variants remain explicit throughout learning state and content:
`de-DE` (Germany), `es-ES` (Spain), `es-419` (Latin America), `zh-Hans` (Simplified Chinese), and
`zh-Hant` (Traditional Chinese). Difficulty is stored twice on purpose: a broad CEFR `level`
(`A1`-`C2`) and a fine `difficulty` value from 0 to 1.

### Code map

The backend keeps immutable input, derived learning state, and orchestration separate:

- `schemas.py` and `models.py` define external contracts and durable rows.
- `learning.py` is the compatibility façade for lesson import, event recording, profile-facing
  operations, and generation requests.
- `lesson_content.py` is the canonical typed view of stored lesson JSON and owns shared sentence,
  run, and token traversal.
- `lesson_activity.py` projects append-only interactions and queue actions into completion, skip,
  exposure, and status facts; `lesson_queue.py` owns the ready/unread queue definitions built from
  that projection.
- `lexeme_learning.py`, `character_learning.py`, `grammar.py`, and `proficiency.py` independently
  replay append-only evidence into disposable derived tables. `character_word_support.py` contains
  the small one-way Chinese candidate-order hint and cannot mutate either reducer.
- `reading_evidence.py`, `spaced_repetition.py`, and `learning_policy.py` hold the small validated
  policies shared by those reducers and read-side projections.
- `language_packs.py`, `language_rules/`, and `learning_units.py` contain language-specific behavior
  and stable vocabulary identity rules.
- `agent_worker.py` coordinates provider-neutral generation stages; the stage callback boundary and
  its artifacts live in the adjacent `generation_*` modules. `generation_tasks.py` alone defines
  durable task kinds, modes, requested topics, and generated lesson keys.

On the frontend, `app.ts` owns navigation and browser-side orchestration. Domain renderers and pure
presentation logic live beside it in focused modules such as `statistics-view.ts`. Closed runtime
values are declared once in `contracts.ts`; language-tag, review-due, search, and term-band rules
live in their named pure modules. New behavior should enter the narrowest domain module;
compatibility imports in a façade are preferable to making reducers import orchestration.

## Automatic generation

`pnpm dev` starts automatic queue maintenance by default, keeping three unread lessons ready—the
current text and two ahead:

```bash
pnpm dev
```

This starts one durable worker alongside the reader. It snapshots the generation brief, closes the
database transaction, runs a read-only callback, validates the returned drafts, and imports them on
the host. The development launcher supervises generation in fresh worker processes, restarting an
unexpected exit with bounded backoff. Relevant Python or language/grammar TOML changes never
interrupt an active callback; the next worker starts with the updated source after that task ends.
The built-in callback is an ephemeral local `codex exec`; it never receives write access. Each
ordinary task first writes frozen learning-language prose. One batched translation call then
overlaps lexical jobs that tokenize batches of sentences (8 by default,
`ARC_LANG_LEXICAL_BATCH_SIZE`; each result stays sentence-local and is validated on its own); at
most three callbacks run at once, of which at most two are lexical. Per-sentence calls repeated a
large fixed prompt for every sentence and cost most of a lesson's tokens. Grammar follows the fully assembled and validated lexical runs so its ranges have
stable anchors. The host scopes model-authored term keys to one sentence, restores established
identities, merges exact cross-sentence identities, and invokes a small lexical reconciliation only
for same-lemma definitions that code cannot safely identify as one sense. It then merges the
four compact stage results and rejects changed source text, missing or reordered keys, invalid
ranges, and cross-stage drift. Calibration remains a single complete callback. After each lesson
the worker takes a fresh learning snapshot for the next slot. A provider-neutral maintenance sweep
repairs missed queue updates every 30 seconds.
Each invalid sentence from a batch gets one immediate sentence-local repair with the exact host
diagnostic, and a sentence missing from a batch (or a failed batch) gets a fresh single-sentence
attempt; valid siblings are checkpointed and reused if the durable task itself must retry. Identity reconciliation
is split into independently validated bounded chunks when a lesson has more ambiguity groups than
one callback can accept; chunks may overlap within the same two lexical lanes and are globally
validated again before assembly. Every other failed DAG stage likewise gets at most one repair
attempt. Every already-validated independent stage is materialized into the next attempt, and only
a failed unit or stage plus its dependents reruns. Ordinary staged work also has an absolute
six-attempt safety cap, enough for one failure in every stage followed by success; legacy calibration
retains its two-attempt cap. A terminal failure remains visible and maintenance does not loop on it;
a fresh accepted reading interaction or skip action can authorize one idempotent successor.
Maintenance compares durable relevant-event and skipped-action high-water marks, so it also recovers
if the process stops between committing evidence and waking the queue. A worker interruption uses
the same bounded failure budget and remains available as retry context. Tasks and callback logs are
inspectable with the `lang generation` commands above. Each task attempt also appends a private
`timings.jsonl` beside `callback.log`, recording callback duration, queue wait, request/response
bytes, available token usage, conflict-group counts, and host preparation, assembly, import, and
total-task spans without copying lesson content into the timing records.

Inactive profiles are skipped during queue maintenance and task claiming, so merely provisioning a
language never invokes an agent callback. If this worker is already running, explicit activation can
enqueue the first required text; otherwise the agent can still use the manual lesson workflow.

For an explicitly manual, token-saving run, disable the worker:

```bash
ARC_LANG_AUTO_GENERATE=0 pnpm dev
```

Unknown-level profiles temporarily use a queue target of one, allowing the second calibration text
to adapt to the first. Once placement is rough, the configured ordinary unread queue target applies.

Provider model choices live in the versioned `config/generation.toml`. Each provider owns separate
`prose`, `lexical`, `translation`, and `grammar` routes with a model and reasoning effort; the
built-in Codex adapter consumes the `codex` table, and prose and lexical tasks use Astra. The worker loads and validates
the provider's complete route set once, before claiming any task, then keeps that immutable snapshot
for its lifetime. A future adapter can consume its own provider table, while a command adapter may
route the same provider-neutral task names independently.

Codex environment variables are emergency overrides only. For either setting, the precedence is
`ARC_LANG_CODEX_TASK_MODEL` or `ARC_LANG_CODEX_TASK_REASONING_EFFORT`, then the corresponding global
`ARC_LANG_CODEX_MODEL` or `ARC_LANG_CODEX_REASONING_EFFORT`, then tracked TOML. Replace `TASK` with
`PROSE`, `LEXICAL`, `TRANSLATION`, or `GRAMMAR`. Legacy one-pass calibration uses the prose route;
`ARC_LANG_CODEX_COMPLETE_*` remains its most-specific compatibility override, followed by
`ARC_LANG_CODEX_PROSE_*`, the global override, and the tracked prose route.

For a host with the shared LLM adapter, set `ARC_LANG_AGENT_CALLBACK=llm` and
`ARC_LANG_AGENT_ROLE=lang-generate`. The role selects its model, harness and account; Lang sends the
same schema-bound read-only stage request. `ARC_LANG_ADMISSION_COMMAND` may name a host admission
command: only exit 0 permits a claim or preference callback, and errors/timeouts defer. Observations
are cached for at most 60 seconds. This replaces the obsolete `ARC_LANG_DEFER_COMMAND` integration.

If the LLM adapter returns its documented unavailable exit (3) after admission, the same task stays
pending with a 60-second retry delay. Already validated stage artifacts survive. Dispatch numbers
(`attempts`) keep increasing to preserve separate artifact directories, while `admission_deferrals`
in the task payload discounts confirmed waits from the draft-attempt cap. Ordinary errors and
timeouts still consume that cap; availability never authorizes retrying an uncertain tool effect.
Preference updates also retain their pending input when the role is unavailable.

For another local agent, set `ARC_LANG_AGENT_CALLBACK=command` and
`ARC_LANG_AGENT_COMMAND='agent-command --flags'`. The command is executed as an argv list without a
shell. It receives canonical requests with `schema_version`, provider-neutral `stage`, and `task`
fields on stdin. Prose, sentence-local lexical annotation, optional lexical conflict reconciliation,
translation, and grammar each have a strict compact result schema; both lexical request kinds keep
the same provider-neutral `lexical` task name. Legacy calibration returns the complete lesson shape.
A callback adapter must permit bounded lexical and translation calls to run concurrently. Stable
environment fields are
`ARC_LANG_PROTOCOL`, `ARC_LANG_JOB_ID`, `ARC_LANG_REQUEST_SHA256`, and the request/response path
fields prefixed with `ARC_LANG_CALLBACK_`; `ARC_LANG_CALLBACK_STAGE` identifies the current phase.
`ARC_LANG_PROFILE` and `ARC_LANG_WORKSPACE` identify the task scope. The host owns lesson keys,
languages, metadata, database writes, and queue reconciliation. `ARC_LANG_JOB_ID` is globally unique
in the form `PROFILE_ID:TASK_ID`; the local numeric task ID remains available as
`ARC_LANG_GENERATION_TASK_ID`.

The callback contract deliberately separates text coverage from curriculum choices. The title and
every body lexical token in a generated draft must be a term-bearing run with an approximate corpus
`frequency_rank`; proper names may leave the rank null and remain context-only. `title_sentence`
contains the clickable title, an exact text match, a unique sentence key, and its translation. Plain
runs may contain only whitespace and punctuation. `target_term_keys` is an advisory teaching subset
reconciled against the request's dynamic candidate policy and body terms actually present;
calibration requests carry their own 8-20 authored probe subset. The host rejects sparse annotation
and conflicting definitions for a stable term key. Manual lesson imports remain backward compatible
and may omit title annotation, full body annotation, or frequency ranks.

At import time, callback keys are provisional. A reference to a known key with the same normalized
lemma is restored to the complete library definition. If the key is already owned by a different
lemma, the host assigns the new definition a deterministic collision-safe key and rewrites run,
target, and calibration references. This also checks stored display-only terms, not just Words/SRS
entries, so an old non-learning span cannot make a later generated lesson fail at import.

For ordinary lessons, the host also owns the numeric challenge and requested length. The prompt
states both explicitly; drafts more than 0.05 away from the requested difficulty or below 60% of the
requested body lexical length are rejected with exact actual/requested diagnostics. Accepted lesson
metadata preserves the provider's declaration but the displayed/adaptive difficulty is the host
target, preventing a model from making easy prose harder merely by relabelling it.

Each request also carries the resolved language pack's generation guidance. Generated drafts must
use reusable language-aware learning units; a contextual `run.pronunciation` may override the
canonical dictionary pronunciation for one surface occurrence without changing the stable term
definition.

Requests also carry the complete resolved grammar catalog and a smaller ranked `priority_grammar`
choice pool. That pool is not a checklist: its preferred and maximum opportunity counts adapt to
text length, placement confidence, difficulty, calibration, and grammar feedback, and zero deliberate
uses is valid when none fit naturally. The callback must nevertheless annotate every clear catalog
construction that appears incidentally, using precise run ranges. The host records offered,
realized, and annotated grammar keys in lesson metadata so skipped choices can cool instead of being
forced into the next text.

Ordinary callback requests also carry one deterministic `content_plan`: a coherent discourse form,
perspective, concrete seed, progression, and ending shape selected to avoid recent plans. Recent
lesson openings and endings are included as short negative examples, while stale target/grammar job
bookkeeping and old calibration probe arrays are omitted from the brief. Linguistic difficulty
controls vocabulary and syntax, not whether the content may be specific or structurally varied. The
callback chooses the situation before selecting optional SRS terms, and the chosen plan is stored in
lesson metadata so later requests can rotate away from it. Calibration retains its narrower probe
contract without a content plan.

Relevant runtime settings are `ARC_LANG_QUEUE_TARGET` (1-10, default 3),
`ARC_LANG_AGENT_MAINTENANCE_SECONDS` (1-3600, default 30),
`ARC_LANG_AGENT_TIMEOUT_SECONDS` (default 600), and `ARC_LANG_AGENT_CALLBACK` (`codex`, `llm` or
`command`). Codex model and effort defaults come from the tracked task routes above; the
`ARC_LANG_CODEX_*` forms are optional emergency overrides, not defaults.

## Content model direction

Keep independent axes instead of one flat genre list:

- discourse form: personal recount, story, article or explainer, dialogue, diary or message;
- fictionality: factual, fictional, or simulated real life;
- genre or tone: science fiction, mystery, slice of life, humour;
- topic or domain: daily life, history, food, technology;
- continuity: standalone, episodic, or serialized.

A normal “my day” text is usually a personal recount with a daily-life topic and slice-of-life tone.
It can be standalone or part of an episodic diary. Declared choices, agent inferences, and recent
feedback adjustments should retain their provenance instead of becoming one flat interest list.

A series should have a stable key, discourse form, fictionality, genres, premise, recurring cast or
concepts, compact continuity summary, open threads, and next episode number. Lessons reference that
key and store an episode summary; generation receives the compact series state rather than every
previous text.
Wordlists and explicit review should consume the same lexeme state and event history as reading.

## Scope

The first version covers reading, term/sentence/grammar/full translations, replayable vocabulary and
grammar adaptation, agent briefs, lesson import, a text library with explicit no-evidence previews,
topic requests, and basic cached TTS. Text series will become collections inside Texts rather than a
separate rail item. Explicit quizzes, richer grammar diagnostics, synchronized sentence audio, and
deeper progress views remain future work.
