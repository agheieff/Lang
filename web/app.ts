import {
  type ProfileActivation,
  type ProfileActivationUpdate,
  parseInterestInput,
  parseProfileActivation,
  QUESTIONNAIRE_CONFIDENCES,
  type QuestionnaireConfidence,
  STARTING_POINTS,
  type StartingPoint,
} from "./activation.js";
import { ActiveTimer, TOUCH_READING_ACTIVITY_WINDOW_MS } from "./active-time.js";
import { appPath } from "./base-path.js";
import {
  type CharactersPayload,
  type CharacterView,
  parseCharacters,
} from "./characters.contract.js";
import {
  CHARACTER_SORTS,
  type CharacterSort,
  type CharacterSortState,
  characterSortSelection,
  DEFAULT_CHARACTER_SORT,
  nextCharacterSort,
  parseCharacterSortSelection,
  reverseCharacterSortResult,
  selectCharacters,
  summarizeCharacters,
} from "./characters.js";
import {
  type EventType,
  type GrammarCatalogEntry,
  type JsonRecord,
  type LearningEvent,
  type Lesson,
  parseReader,
  parseWords,
  type ReaderPayload,
  type ReaderProgress,
  type Run,
  type Sentence,
  type TermBand,
  type WordsPayload,
  type WordView,
} from "./contracts.js";
import {
  ActiveTimeStore,
  activeTimeStorageKey,
  DurableEventStore,
  eventQueueStorageKey,
  flushStoredEvents,
  LessonSessionStore,
  lessonSessionStorageKey,
  migrateLegacyPersistence,
} from "./contracts.persistence.js";
import { isDialogueSentence } from "./dialogue-format.js";
import { byId, plural, textElement } from "./dom.js";
import {
  CONSTRUCTION_STATUS_DETAILS,
  constructionStatus,
  DEFAULT_GRAMMAR_SORT,
  formatGrammarDue,
  formatGrammarStatus,
  GRAMMAR_SORTS,
  type GrammarConstruction,
  type GrammarSort,
  type GrammarState,
  parseGrammar,
  selectGrammar,
} from "./grammar.js";
import { visibleGrammarOccurrences } from "./grammar-markers.js";
import { isChineseLanguage } from "./language-tag.js";
import { formatLibraryDate, formatLibraryDateTime } from "./library-date.js";
import { renderHanziTones, renderPinyin } from "./pinyin.js";
import { copyReaderSelection } from "./reader-copy.js";
import { readerLevelPresentation } from "./reader-level.js";
import {
  parseReaderLaunch,
  type ReaderLaunch,
  type ReaderMode,
  readerApiPath,
  readerModeRecordsEvidence,
  textActionHref,
} from "./reader-mode.js";
import {
  contentExperiment,
  parseReadingPreferences,
  type ReadingPreferences,
} from "./reading-preferences.js";
import {
  LONG_PRESS_MOVE_TOLERANCE_PX,
  SENTENCE_LONG_PRESS_MS,
  SentenceRevealState,
  shouldHandleSentenceClick,
  TOUCH_SENTENCE_HELP_GRACE_MS,
} from "./reveal-state.js";
import { pronunciationForRun } from "./run-pronunciation.js";
import { parseStatistics, type StatisticsPayload } from "./statistics.js";
import { renderStatisticsView } from "./statistics-view.js";
import { attachTermBand } from "./term-band.js";
import {
  setTextCardActionPending,
  type TextDisposition,
  textDispositionPath,
  textDispositionPayload,
} from "./text-disposition.js";
import {
  orderedReadyTexts,
  queueNeighborLessonId,
  readyQueuePositionLabel,
  type TextQueueMoveDirection,
  textQueueMovePath,
  textQueueMovePayload,
} from "./text-queue.js";
import {
  parseTextLibrary,
  parseTextPreview,
  parseTextRequest,
  type TextLibraryItem,
  type TextLibraryPayload,
  type TextPreparationItem,
  type TextRequestItem,
  type TextStatus,
} from "./texts.contract.js";
import {
  applyTheme,
  parseThemeMode,
  persistThemePreference,
  readThemePreference,
  type ThemeMode,
} from "./theme.js";
import { ToneColorPreference, toneColorStorageKey } from "./tone-color.js";
import {
  audioStatusFetchFailure,
  audioStatusMessage,
  browserFallbackMessage,
  browserSpeechFailure,
  chooseBrowserVoice,
  type LessonAudioStatus,
  lessonSpeechChunks,
  localAudioElementFailure,
  localAudioPlayFailure,
  parseLessonAudioStatus,
  parseTtsSettings,
  persistPlaybackRate,
  readPlaybackRate,
  type TtsSettings,
} from "./tts.js";
import { splitClosingPunctuation } from "./typography.js";
import {
  DEFAULT_WORD_SORT,
  nextWordSort,
  parseWordSortSelection,
  reverseWordSortResult,
  type WordSortState,
  wordSortSelection,
} from "./word-sort-ui.js";
import { formatDue, selectWords, WORD_SORTS, type WordSort } from "./words.js";

const loadingView = byId("loading-view");
const errorView = byId("error-view");
const activationView = byId("activation-view");
const readerView = byId("reader-view");
const textsView = byId("texts-view");
const wordsView = byId("words-view");
const charactersView = byId("characters-view");
const grammarView = byId("grammar-view");
const statisticsView = byId("statistics-view");
const settingsView = byId("settings-view");
const preferencesText = byId<HTMLTextAreaElement>("preferences-text");
const preferencesMessage = byId<HTMLTextAreaElement>("preferences-message");
const preferencesSave = byId<HTMLButtonElement>("preferences-save");
const preferencesSend = byId<HTMLButtonElement>("preferences-send");
const preferencesStatus = byId("preferences-status");
let preferencesRevision: number | null = null;
const lessonContent = byId<HTMLElement>("lesson-content");
const fullPanel = byId("full-translation");
const fullContent = byId("full-translation-content");
const fullButton = byId<HTMLButtonElement>("full-translation-button");
const completeButton = byId<HTMLButtonElement>("complete-button");
const skipLessonButtons = [
  byId<HTMLButtonElement>("skip-lesson-button"),
  byId<HTMLButtonElement>("skip-lesson-completion-button"),
];
const actionStatusMessage = byId("action-status-message");
const statusMessage = byId("status-message");
const popover = byId("gloss-popover");
const sentencePopover = byId("sentence-translation-popover");
const sentencePopoverSource = byId("sentence-source-text");
const sentencePopoverText = byId("sentence-translation-text");
const grammarPopover = byId("grammar-popover");
const grammarPopoverContent = byId("grammar-popover-content");
const sentenceHelpModeButton = byId<HTMLButtonElement>("sentence-help-mode");
const feedbackDetails = byId<HTMLDetailsElement>("expert-feedback");
const profileSwitch = byId<HTMLSelectElement>("profile-switch");
const toneColorToggle = byId<HTMLButtonElement>("tone-color-toggle");
const toneColorState = byId("tone-color-state");
const lessonAudio = byId<HTMLAudioElement>("lesson-audio");
const lessonAudioButton = byId<HTMLButtonElement>("lesson-audio-button");
const lessonAudioSpeed = byId<HTMLSelectElement>("lesson-audio-speed");
const lessonAudioStatus = byId("lesson-audio-status");
const ttsVoiceSelect = byId<HTMLSelectElement>("tts-voice-select");
const ttsSettingsStatus = byId("tts-settings-status");
const themeColorMeta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]');
const rawProfileId = document.body.dataset.profileId;
if (!rawProfileId) throw new Error("Missing active profile ID");
const profileId: string = rawProfileId;
lessonAudioSpeed.value = String(readPlaybackRate(window.localStorage, profileId));
type InitialView =
  | "characters"
  | "grammar"
  | "reading"
  | "settings"
  | "statistics"
  | "texts"
  | "words";
const requestedInitialView = document.body.dataset.initialView;
const initialView: InitialView =
  requestedInitialView === "words" ||
  requestedInitialView === "characters" ||
  requestedInitialView === "grammar" ||
  requestedInitialView === "statistics" ||
  requestedInitialView === "settings" ||
  requestedInitialView === "texts"
    ? requestedInitialView
    : "reading";
const VIEW_ERROR_COPY: Partial<Record<InitialView, { eyebrow: string; heading: string }>> = {
  characters: {
    eyebrow: "Characters unavailable",
    heading: "We couldn’t load your character exposure.",
  },
  grammar: {
    eyebrow: "Grammar unavailable",
    heading: "We couldn’t load your grammar evidence.",
  },
  settings: {
    eyebrow: "Settings unavailable",
    heading: "We couldn’t load your settings.",
  },
  statistics: {
    eyebrow: "Statistics unavailable",
    heading: "We couldn’t load your reading statistics.",
  },
  texts: {
    eyebrow: "Texts unavailable",
    heading: "We couldn’t load your text library.",
  },
  words: {
    eyebrow: "Words unavailable",
    heading: "We couldn’t load your vocabulary.",
  },
};
const profileApi = appPath(`/api/profiles/${encodeURIComponent(profileId)}`);
const toneColorPreference = new ToneColorPreference(
  window.localStorage,
  toneColorStorageKey(profileId),
);

function setTheme(theme: ThemeMode, announce = false): void {
  applyTheme(document.documentElement, theme);
  if (themeColorMeta) themeColorMeta.content = theme === "dark" ? "#111714" : "#f4f1e9";
  for (const input of document.querySelectorAll<HTMLInputElement>('input[name="color-theme"]')) {
    input.checked = input.value === theme;
  }
  byId("theme-status").textContent = announce
    ? `${theme === "dark" ? "Dark" : "Light"} theme applied.`
    : "";
}

setTheme(readThemePreference(window.localStorage));

const feedbackConflicts: Record<string, string> = {
  shorter: "longer",
  longer: "shorter",
  easier: "more_challenging",
  more_challenging: "easier",
  more_grammar: "less_grammar",
  less_grammar: "more_grammar",
  same_topic: "new_topic",
  new_topic: "same_topic",
};
for (const input of feedbackDetails.querySelectorAll<HTMLInputElement>(
  'input[name="lesson-feedback"]',
)) {
  input.addEventListener("change", () => {
    const conflict = feedbackConflicts[input.value];
    if (!input.checked || !conflict) return;
    const opposite = feedbackDetails.querySelector<HTMLInputElement>(
      `input[name="lesson-feedback"][value="${conflict}"]`,
    );
    if (opposite) opposite.checked = false;
  });
}

let lesson: Lesson | null = null;
let lessonId: number | null = null;
let sessionId = "";
let rating: -1 | 1 | null = null;
let glossAnchor: HTMLButtonElement | null = null;
let sentenceHelpAnchor: HTMLButtonElement | null = null;
let sentenceHelpUnit: HTMLElement | null = null;
let pendingSentenceEvidence: number | null = null;
let grammarHelpAnchor: HTMLButtonElement | null = null;
let sentenceReveals = new SentenceRevealState();
let revealedGrammarOccurrences = new Set<string>();
let readerGrammarCatalog = new Map<string, GrammarCatalogEntry>();
let comfortableGrammarConstructionKeys = new Set<string>();
let sentenceSelectionMode = false;
let fullTranslationRevealed = false;
let lessonPollTimer: number | null = null;
let textsPollTimer: number | null = null;
let wordsPayload: WordsPayload | null = null;
let charactersPayload: CharactersPayload | null = null;
let grammarPayload: GrammarState | null = null;
let textsPayload: TextLibraryPayload | null = null;
let currentLearningLanguage = "";
let wordSortState: WordSortState = DEFAULT_WORD_SORT;
let characterSortState: CharacterSortState = DEFAULT_CHARACTER_SORT;
let grammarSort: GrammarSort = DEFAULT_GRAMMAR_SORT;
let readerMode: ReaderMode = "standard";
let currentAudioStatus: LessonAudioStatus | null = null;
let audioPollTimer: number | null = null;
let audioRequestVersion = 0;
let browserSpeechToken = 0;
let browserSpeechActive = false;
let actionStatusTimer: number | null = null;

class EventQueue {
  private timer: number | null = null;
  private flushing: Promise<void> | null = null;

  constructor(private readonly store: DurableEventStore) {}

  add(event: LearningEvent, immediate = false): void {
    this.store.append(event);
    if (this.timer !== null) window.clearTimeout(this.timer);
    this.timer = window.setTimeout(
      () => {
        this.timer = null;
        void this.flush().catch(showSyncWarning);
      },
      immediate ? 0 : 450,
    );
  }

  flush(): Promise<void> {
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    if (!this.flushing) {
      this.flushing = this.drain().finally(() => {
        this.flushing = null;
      });
    }
    return this.flushing;
  }

  clear(): void {
    if (this.timer !== null) {
      window.clearTimeout(this.timer);
      this.timer = null;
    }
    this.store.clear();
  }

  sendOnExit(): void {
    const pending = this.store.snapshot();
    if (pending.length === 0) return;
    const body = JSON.stringify({ events: pending });
    navigator.sendBeacon(`${profileApi}/events`, new Blob([body], { type: "application/json" }));
  }

  private async drain(): Promise<void> {
    while (this.store.snapshot().length > 0) {
      await flushStoredEvents(this.store, (batch) => this.send(batch));
    }
  }

  private async send(events: readonly LearningEvent[]): Promise<void> {
    const response = await fetch(`${profileApi}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ events }),
    });
    if (!response.ok) throw new Error(`Event sync failed (${response.status})`);
    for (const event of events) {
      if (event.type === "lesson.completed") {
        lessonSessions.complete(event.lesson_id, event.session_id);
        activeTimer.complete(event.lesson_id, event.session_id);
      }
    }
  }
}

migrateLegacyPersistence(window.localStorage, profileId);
const lessonSessions = new LessonSessionStore(
  window.localStorage,
  lessonSessionStorageKey(profileId),
);
const events = new EventQueue(
  new DurableEventStore(window.localStorage, eventQueueStorageKey(profileId)),
);
const activeTimeStore = new ActiveTimeStore(window.localStorage, activeTimeStorageKey(profileId));
const activeTimer = new ActiveTimer(activeTimeStore, {
  clock: () => performance.now(),
  isHidden: () => document.hidden,
  render: (activeMs) => {
    const seconds = Math.floor(activeMs / 1_000);
    const minutes = Math.floor(seconds / 60);
    byId("active-time").textContent =
      `Active reading · ${minutes}:${String(seconds % 60).padStart(2, "0")}`;
  },
});
const coarsePointer = window.matchMedia("(pointer: coarse)");
window.addEventListener(
  "pointermove",
  (event) =>
    activeTimer.noteMovement(
      event.pointerType === "touch" ? TOUCH_READING_ACTIVITY_WINDOW_MS : undefined,
    ),
  { passive: true },
);
window.addEventListener(
  "pointerdown",
  (event) => {
    if (event.pointerType === "touch") activeTimer.noteMovement(TOUCH_READING_ACTIVITY_WINDOW_MS);
  },
  { passive: true },
);
window.addEventListener(
  "scroll",
  () => {
    // Touch scrolling cancels pointer events, so scroll itself is the reading signal there.
    if (coarsePointer.matches) activeTimer.noteMovement(TOUCH_READING_ACTIVITY_WINDOW_MS);
  },
  { passive: true },
);
document.addEventListener("visibilitychange", () => activeTimer.visibilityChanged());
window.setInterval(() => activeTimer.tick(), 1_000);

function now(): string {
  return new Date().toISOString();
}

function makeEvent(type: EventType, payload: JsonRecord): LearningEvent {
  if (!readerModeRecordsEvidence(readerMode)) {
    throw new Error("Preview mode cannot create learning events");
  }
  if (lessonId === null || !sessionId) throw new Error("No active lesson session");
  return {
    event_id: crypto.randomUUID(),
    session_id: sessionId,
    lesson_id: lessonId,
    type,
    occurred_at: now(),
    payload,
  };
}

function recordLearningEvent(type: EventType, payload: JsonRecord, immediate = false): void {
  if (!readerModeRecordsEvidence(readerMode)) return;
  events.add(makeEvent(type, payload), immediate);
}

function summaryStat(className: string, value: number, label: string): HTMLElement {
  const item = document.createElement("span");
  item.className = className;
  item.append(textElement("strong", "", value.toLocaleString()), document.createTextNode(label));
  return item;
}

function showActionStatus(message: string): void {
  if (actionStatusTimer !== null) window.clearTimeout(actionStatusTimer);
  actionStatusMessage.textContent = message;
  actionStatusMessage.hidden = false;
  actionStatusTimer = window.setTimeout(() => {
    actionStatusTimer = null;
    actionStatusMessage.hidden = true;
  }, 5_000);
}

function setView(
  view:
    | "loading"
    | "error"
    | "activation"
    | "characters"
    | "reader"
    | "settings"
    | "statistics"
    | "texts"
    | "words"
    | "grammar",
): void {
  loadingView.hidden = view !== "loading";
  errorView.hidden = view !== "error";
  activationView.hidden = view !== "activation";
  readerView.hidden = view !== "reader";
  textsView.hidden = view !== "texts";
  wordsView.hidden = view !== "words";
  charactersView.hidden = view !== "characters";
  grammarView.hidden = view !== "grammar";
  statisticsView.hidden = view !== "statistics";
  settingsView.hidden = view !== "settings";
  if (view !== "reader") sentenceHelpModeButton.hidden = true;
  if (view !== "texts") clearTextsPoll();
  if (view !== "reader") {
    closeGloss();
    closeSentenceHelp();
    closeGrammarHelp();
    clearAudioPoll();
    audioRequestVersion += 1;
    stopLessonAudio(true);
  }
}

async function getJson(
  path: string,
  failure: string | ((status: number) => Error),
): Promise<unknown> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw typeof failure === "string"
      ? new Error(`${failure} returned ${response.status}`)
      : failure(response.status);
  }
  return response.json();
}

function postJson(path: string, body: unknown, method = "POST"): Promise<Response> {
  return fetch(path, {
    method,
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
}

async function loadActivation(): Promise<ProfileActivation> {
  return parseProfileActivation(await getJson(`${profileApi}/activation`, "Activation status"));
}

function renderActivation(): void {
  setView("activation");
  byId("activation-status").textContent = "";
}

async function submitActivation(): Promise<void> {
  const selected = document.querySelector<HTMLInputElement>('input[name="starting-point"]:checked');
  const startingPoint = selected?.value;
  if (!startingPoint || !STARTING_POINTS.includes(startingPoint as StartingPoint)) {
    throw new Error("Choose a reading starting point.");
  }
  const confidenceValue = byId<HTMLSelectElement>("activation-confidence").value;
  if (!QUESTIONNAIRE_CONFIDENCES.includes(confidenceValue as QuestionnaireConfidence)) {
    throw new Error("Choose how confident you are in the estimate.");
  }
  const textLengthValue = byId<HTMLSelectElement>("activation-text-length").value;
  const textLength = textLengthValue ? Number(textLengthValue) : undefined;
  const payload: ProfileActivationUpdate = {
    starting_point: startingPoint as StartingPoint,
    confidence: confidenceValue as QuestionnaireConfidence,
    interests: parseInterestInput(byId<HTMLInputElement>("activation-interests").value),
    ...(textLength === undefined ? {} : { text_length: textLength }),
  };
  const submit = byId<HTMLButtonElement>("activation-submit");
  const status = byId("activation-status");
  submit.disabled = true;
  submit.textContent = "Starting…";
  status.textContent = "Saving your starting point…";
  try {
    const response = await postJson(`${profileApi}/activation`, payload);
    if (!response.ok) throw new Error(`Activation returned ${response.status}`);
    const activated = parseProfileActivation(await response.json());
    if (!activated.active) throw new Error("The language profile was not activated.");
    window.location.reload();
  } catch (error) {
    status.textContent = error instanceof Error ? error.message : "Activation failed.";
    submit.disabled = false;
    submit.textContent = "Start this language";
  }
}

function configureToneColors(learningLanguage: string): void {
  currentLearningLanguage = learningLanguage;
  const available = isChineseLanguage(learningLanguage);
  const enabled = available && toneColorPreference.enabled();
  toneColorToggle.hidden = !available;
  toneColorToggle.setAttribute("aria-pressed", String(enabled));
  toneColorToggle.setAttribute(
    "aria-label",
    enabled ? "Turn Hanzi tone colors off" : "Color Chinese characters by tone",
  );
  toneColorToggle.title = enabled ? "Turn Hanzi tone colors off" : "Color Hanzi by tone";
  toneColorState.textContent = enabled ? "On" : "Off";
  document.body.dataset.hanziToneColors = String(enabled);
}

function browserSpeechAvailable(): boolean {
  return "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
}

function audioPlaybackRate(): number {
  const value = Number(lessonAudioSpeed.value);
  return Number.isFinite(value) && value >= 0.5 && value <= 2 ? value : 1;
}

function browserSpeechFallback(learningLanguage: string) {
  const available = browserSpeechAvailable();
  const voice = available
    ? chooseBrowserVoice(window.speechSynthesis.getVoices(), learningLanguage)
    : null;
  return {
    available,
    learningLanguage,
    matchingVoiceName: voice?.name ?? null,
  };
}

class AudioStatusRequestError extends Error {
  constructor(readonly status: number) {
    super(`Audio status returned ${status}`);
  }
}

function setAudioButton(playing: boolean): void {
  byId("lesson-audio-button-icon").textContent = playing ? "❚❚" : "▶";
  byId("lesson-audio-button-label").textContent = playing ? "Pause" : "Listen";
  lessonAudioButton.setAttribute(
    "aria-label",
    playing ? "Pause reading this text aloud" : "Listen to this text",
  );
}

function clearAudioPoll(): void {
  if (audioPollTimer === null) return;
  window.clearTimeout(audioPollTimer);
  audioPollTimer = null;
}

function stopBrowserSpeech(): void {
  browserSpeechToken += 1;
  if (browserSpeechAvailable()) window.speechSynthesis.cancel();
  browserSpeechActive = false;
}

function stopLessonAudio(resetPosition = false): void {
  stopBrowserSpeech();
  lessonAudio.pause();
  if (resetPosition && lessonAudio.readyState > 0) lessonAudio.currentTime = 0;
  setAudioButton(false);
}

function resetLessonAudio(): number {
  clearAudioPoll();
  stopLessonAudio(true);
  audioRequestVersion += 1;
  currentAudioStatus = null;
  lessonAudio.removeAttribute("src");
  lessonAudio.load();
  lessonAudioButton.disabled = true;
  lessonAudioStatus.textContent = "Checking listening audio…";
  return audioRequestVersion;
}

async function configureLessonAudio(selectedLessonId: number): Promise<void> {
  const requestVersion = resetLessonAudio();
  await refreshLessonAudio(selectedLessonId, requestVersion);
}

async function refreshLessonAudio(selectedLessonId: number, requestVersion: number): Promise<void> {
  try {
    const status = parseLessonAudioStatus(
      await getJson(
        `${profileApi}/lessons/${selectedLessonId}/audio/status`,
        (failureStatus) => new AudioStatusRequestError(failureStatus),
      ),
    );
    if (
      requestVersion !== audioRequestVersion ||
      lessonId !== selectedLessonId ||
      status.lesson_id !== selectedLessonId
    ) {
      return;
    }
    currentAudioStatus = status;
    const fallback =
      status.state === "disabled"
        ? {
            available: false,
            learningLanguage: lesson?.learning_language ?? "this language",
            matchingVoiceName: null,
          }
        : browserSpeechFallback(lesson?.learning_language ?? "this language");
    if (status.state === "ready" && status.audio_url) {
      lessonAudio.src = status.audio_url;
      lessonAudio.load();
      lessonAudioButton.disabled = false;
      lessonAudioStatus.textContent = audioStatusMessage(status, fallback);
      return;
    }
    lessonAudioButton.disabled = !fallback.available;
    lessonAudioStatus.textContent = audioStatusMessage(status, fallback);
    if (status.state === "preparing") {
      audioPollTimer = window.setTimeout(() => {
        audioPollTimer = null;
        void refreshLessonAudio(selectedLessonId, requestVersion);
      }, 5_000);
    }
  } catch (error) {
    if (requestVersion !== audioRequestVersion || lessonId !== selectedLessonId) return;
    currentAudioStatus = null;
    const fallback = browserSpeechFallback(lesson?.learning_language ?? "this language");
    lessonAudioButton.disabled = !fallback.available;
    const detail =
      error instanceof AudioStatusRequestError
        ? audioStatusFetchFailure(error.status)
        : error instanceof TypeError
          ? audioStatusFetchFailure(null)
          : "The local app returned an invalid audio status. Refresh the page to try again.";
    lessonAudioStatus.textContent = `${detail} ${browserFallbackMessage(fallback)}`;
  }
}

function speakWithBrowser(localFailure: string | null = null): void {
  if (!lesson) return;
  if (currentAudioStatus?.state === "disabled") {
    lessonAudioStatus.textContent = "Activate this language before using listening audio.";
    return;
  }
  if (!browserSpeechAvailable()) {
    lessonAudioStatus.textContent = `${
      localFailure ? `${localFailure} ` : ""
    }No browser speech engine is available here.`;
    return;
  }
  stopBrowserSpeech();
  const token = browserSpeechToken;
  const chunks = lessonSpeechChunks(lesson);
  const learningLanguage = lesson.learning_language;
  const voice = chooseBrowserVoice(window.speechSynthesis.getVoices(), learningLanguage);
  let index = 0;
  browserSpeechActive = true;
  setAudioButton(true);
  const systemVoiceCopy = voice
    ? `${voice.name} · System voice${voice.localService ? "" : " (may use a network service)"}`
    : `No matching ${learningLanguage} system voice is listed; trying the browser default`;
  lessonAudioStatus.textContent = localFailure
    ? `${localFailure} ${systemVoiceCopy} is playing instead.`
    : systemVoiceCopy;

  const speakNext = (): void => {
    if (token !== browserSpeechToken) return;
    const chunk = chunks[index];
    if (chunk === undefined) {
      browserSpeechActive = false;
      setAudioButton(false);
      return;
    }
    index += 1;
    const utterance = new SpeechSynthesisUtterance(chunk);
    utterance.lang = voice?.lang ?? learningLanguage;
    utterance.rate = audioPlaybackRate();
    if (voice) utterance.voice = voice;
    utterance.onend = speakNext;
    utterance.onerror = (event) => {
      if (token !== browserSpeechToken) return;
      browserSpeechActive = false;
      setAudioButton(false);
      lessonAudioStatus.textContent = browserSpeechFailure(event.error);
    };
    try {
      window.speechSynthesis.speak(utterance);
    } catch (error) {
      browserSpeechActive = false;
      setAudioButton(false);
      const blocked =
        error instanceof DOMException &&
        (error.name === "NotAllowedError" || error.name === "SecurityError");
      lessonAudioStatus.textContent = browserSpeechFailure(
        blocked ? "not-allowed" : "synthesis-failed",
      );
    }
  };
  speakNext();
}

async function toggleLessonAudio(): Promise<void> {
  if (currentAudioStatus?.state === "ready" && lessonAudio.src) {
    stopBrowserSpeech();
    if (!lessonAudio.paused) {
      lessonAudio.pause();
      return;
    }
    lessonAudio.playbackRate = audioPlaybackRate();
    try {
      await lessonAudio.play();
    } catch (error) {
      speakWithBrowser(localAudioPlayFailure(error));
    }
    return;
  }
  if (browserSpeechActive) {
    stopBrowserSpeech();
    setAudioButton(false);
  } else {
    speakWithBrowser();
  }
}

interface LessonRenderState {
  lesson_id: number;
  lesson: Lesson;
  progress: ReaderProgress;
  term_bands: Record<string, TermBand>;
  grammar_catalog?: GrammarCatalogEntry[];
  comfortable_grammar_construction_keys?: string[];
}

function renderReaderMode(mode: ReaderMode, selectedFromLibrary: boolean): void {
  const banner = byId("reader-mode-banner");
  const title = byId("reader-mode-title");
  const detail = byId("reader-mode-detail");
  banner.classList.toggle("reader-mode-banner-preview", mode === "preview");
  banner.classList.toggle("reader-mode-banner-reread", mode === "reread");
  if (mode === "preview") {
    title.textContent = "Peek mode";
    detail.textContent =
      "Nothing here is recorded. Word help and translations are available without changing progress.";
    banner.hidden = false;
  } else if (mode === "reread") {
    title.textContent = "Reading again";
    detail.textContent =
      "This is a fresh session. Reveals and completion update your learning evidence.";
    banner.hidden = false;
  } else if (selectedFromLibrary) {
    title.textContent = "Reading from your library";
    detail.textContent = "Progress is recorded normally, just like in the main reading queue.";
    banner.hidden = false;
  } else {
    banner.hidden = true;
  }
}

function renderLesson(
  state: LessonRenderState,
  mode: ReaderMode = "standard",
  selectedFromLibrary = false,
): void {
  readerMode = mode;
  resetReaderSkipButtons();
  for (const button of skipLessonButtons) {
    button.hidden = mode !== "standard" || state.progress.completed;
  }
  configureToneColors(state.lesson.learning_language);
  clearLessonPoll();
  lesson = state.lesson;
  const experiment = lesson ? contentExperiment(lesson.metadata) : null;
  byId("content-experiment").hidden = experiment === null;
  byId("content-experiment-kind").textContent = experiment?.kind ?? "";
  byId("content-experiment-text").textContent = experiment?.question ?? "";
  lessonId = state.lesson_id;
  if (readerModeRecordsEvidence(mode)) {
    const preferredSession = mode === "reread" ? crypto.randomUUID() : state.progress.session_id;
    sessionId = lessonSessions.getOrCreate(
      lessonId,
      preferredSession,
      crypto.randomUUID.bind(crypto),
    );
  } else {
    sessionId = "";
  }
  rating = null;
  fullTranslationRevealed = mode === "preview" ? false : state.progress.full_translation_revealed;
  closeGloss();
  closeSentenceHelp();
  closeGrammarHelp();
  setSentenceSelectionMode(false);
  sentenceReveals = new SentenceRevealState(
    mode === "preview" ? [] : state.progress.revealed_sentence_keys,
  );
  revealedGrammarOccurrences = new Set(
    mode === "preview" ? [] : state.progress.revealed_grammar_occurrence_keys,
  );
  readerGrammarCatalog = new Map((state.grammar_catalog ?? []).map((entry) => [entry.key, entry]));
  comfortableGrammarConstructionKeys = new Set(state.comfortable_grammar_construction_keys ?? []);
  lessonContent.replaceChildren();
  fullContent.replaceChildren();
  fullPanel.hidden = true;
  fullButton.textContent = "Show full translation";
  completeButton.disabled = !readerModeRecordsEvidence(mode);
  byId("completion-card").hidden = mode === "preview";
  renderReaderMode(mode, selectedFromLibrary);
  feedbackDetails.open = false;
  for (const input of feedbackDetails.querySelectorAll<HTMLInputElement>("input:checked")) {
    input.checked = false;
  }

  const revealedSentences = new Set(state.progress.revealed_sentence_keys);
  const title = byId("lesson-title");
  title.dir = "auto";
  if (lesson.title_sentence) {
    title.replaceChildren(
      renderSentence(
        lesson.title_sentence,
        revealedSentences.has(lesson.title_sentence.key),
        state.term_bands,
      ),
    );
  } else {
    title.textContent = lesson.title;
  }
  const level = readerLevelPresentation(lesson.learning_language, lesson.level, lesson.difficulty);
  const levelDetail = byId("lesson-level-detail");
  levelDetail.textContent = level.label;
  levelDetail.title = level.title ?? "";
  const difficultyDetail = byId("lesson-difficulty-detail");
  difficultyDetail.textContent = `${Math.round(lesson.difficulty * 100)}%`;
  difficultyDetail.title = "Fine-grained adaptive scale used to tune generated texts";
  const topicDetail = byId("lesson-topic-detail");
  const topicDetailItem = byId("lesson-topic-detail-item");
  topicDetail.textContent = lesson.topic ?? "";
  topicDetailItem.hidden = !lesson.topic;
  lessonContent.dir = "auto";
  let hasTranslations = Boolean(lesson.title_sentence?.translation);

  if (lesson.title_sentence?.translation) {
    fullContent.append(
      textElement("p", "translation-paragraph", lesson.title_sentence.translation),
    );
  }

  for (const block of lesson.blocks) {
    const paragraph = document.createElement("p");
    paragraph.className = "lesson-block";
    const translatedParagraph = document.createElement("p");
    translatedParagraph.className = "translation-paragraph";

    for (const sentence of block.sentences) {
      const sourceText = sentence.runs.map((run) => run.text).join("");
      const dialogue = isDialogueSentence(sourceText);
      paragraph.append(
        renderSentence(sentence, revealedSentences.has(sentence.key), state.term_bands, dialogue),
      );
      if (sentence.translation) {
        hasTranslations = true;
        translatedParagraph.append(
          textElement(
            "span",
            dialogue
              ? "translation-sentence translation-sentence-dialogue"
              : "translation-sentence",
            sentence.translation,
          ),
          document.createTextNode(" "),
        );
      }
    }
    lessonContent.append(paragraph);
    fullContent.append(translatedParagraph);
  }

  sentenceHelpModeButton.hidden = !hasTranslations;
  fullButton.hidden = !hasTranslations;
  if (readerModeRecordsEvidence(mode)) {
    renderRatings();
    activeTimer.start(lessonId, sessionId);
  }
  setView("reader");
  void configureLessonAudio(state.lesson_id);
  if (!state.progress.started) recordLearningEvent("lesson.started", {}, true);
}

function renderTexts(state: TextLibraryPayload): void {
  textsPayload = state;
  configureToneColors(state.learning_language);
  const groups: Record<TextStatus, TextLibraryItem[]> = {
    queued: [],
    in_progress: [],
    skipped: [],
    read: [],
  };
  for (const item of state.texts) groups[item.status].push(item);
  const summary = byId("texts-summary");
  const summaryItems = [
    summaryStat("texts-summary-stat", groups.in_progress.length, "in progress"),
    summaryStat("texts-summary-stat", groups.queued.length, "ready"),
    summaryStat("texts-summary-stat", groups.skipped.length, "skipped"),
    summaryStat("texts-summary-stat", groups.read.length, "read"),
  ];
  if (state.preparations.length > 0) {
    summaryItems.unshift(summaryStat("texts-summary-stat", state.preparations.length, "preparing"));
  }
  summary.replaceChildren(...summaryItems);
  renderTextPreparations(state.preparations);
  for (const status of ["in_progress", "queued", "skipped", "read"] as const) {
    renderTextGroup(status, groups[status], state.learning_language);
  }
  renderTextRequests(state.requests);
  setView("texts");
  scheduleTextsPoll(state.preparations, state.requests);
}

const preparationStateLabels: Record<TextPreparationItem["state"], string> = {
  pending: "Queued",
  running: "Writing",
};

function renderTextPreparations(preparations: readonly TextPreparationItem[]): void {
  const group = byId("texts-preparing-group");
  const list = byId("texts-preparing-list");
  group.hidden = preparations.length === 0;
  byId("texts-preparing-count").textContent = String(preparations.length);
  if (preparations.length === 0) {
    list.replaceChildren();
    return;
  }
  list.replaceChildren(...preparations.map(renderTextPreparation));
}

function renderTextPreparation(preparation: TextPreparationItem): HTMLElement {
  const card = document.createElement("article");
  card.className = `text-card text-preparation-card text-preparation-${preparation.state}`;
  card.setAttribute("aria-busy", "true");
  const body = document.createElement("div");
  body.className = "text-card-body";
  const heading = document.createElement("div");
  heading.className = "text-card-heading";
  heading.append(
    textElement("span", "text-status", preparationStateLabels[preparation.state]),
    textElement(
      "h3",
      "",
      preparation.request_kind === "topic_request" ? "Requested text" : "Adaptive text",
    ),
  );
  body.append(heading);
  if (preparation.requested_topic !== null) {
    const topic = textElement("p", "text-card-topic", preparation.requested_topic);
    topic.dir = "auto";
    body.append(topic);
  }
  const meta = document.createElement("div");
  meta.className = "text-card-meta";
  meta.append(
    textElement(
      "span",
      "",
      preparation.state === "running"
        ? "The learning agent is preparing this now"
        : "Waiting for the learning agent",
    ),
    textElement("span", "", `Queued ${formatLibraryDateTime(preparation.created_at)}`),
  );
  body.append(meta);
  card.append(body);
  return card;
}

function renderTextGroup(
  status: TextStatus,
  texts: readonly TextLibraryItem[],
  learningLanguage: string,
): void {
  const suffix = status === "in_progress" ? "progress" : status;
  const selected = status === "queued" ? orderedReadyTexts(texts) : texts;
  byId(`texts-${suffix}-count`).textContent = String(selected.length);
  const list = byId(`texts-${suffix}-list`);
  if (selected.length === 0) {
    const empty = textElement(
      "p",
      "texts-group-empty",
      status === "in_progress"
        ? "No text is currently underway."
        : status === "queued"
          ? "No prepared text is waiting right now."
          : status === "skipped"
            ? "Texts you set aside will remain available here."
            : "Completed texts will collect here.",
    );
    list.replaceChildren(empty);
    return;
  }
  list.replaceChildren(
    ...selected.map((item) =>
      renderTextCard(item, learningLanguage, status === "queued" ? selected : undefined),
    ),
  );
}

function renderTextCard(
  item: TextLibraryItem,
  learningLanguage: string,
  readyTexts?: readonly TextLibraryItem[],
): HTMLElement {
  const card = document.createElement("article");
  card.className = `text-card text-card-${item.status.replace("_", "-")}`;
  card.dataset.lessonId = String(item.id);
  const body = document.createElement("div");
  body.className = "text-card-body";
  const heading = document.createElement("div");
  heading.className = "text-card-heading";
  const queuePosition = readyQueuePositionLabel(item);
  if (queuePosition !== null) {
    const status = textElement("span", "text-status", queuePosition);
    status.setAttribute("aria-label", `Queue position ${queuePosition}`);
    status.title = `Queue position ${queuePosition}`;
    heading.append(status);
  }
  const title = textElement("h3", "", item.title);
  title.dir = "auto";
  heading.append(title);
  body.append(heading);
  if (item.topic) {
    const topic = textElement("p", "text-card-topic", item.topic);
    topic.dir = "auto";
    body.append(topic);
  }

  const level = readerLevelPresentation(learningLanguage, item.level, item.difficulty);
  const meta = document.createElement("div");
  meta.className = "text-card-meta";
  meta.append(
    textElement("span", "", level.label),
    textElement("span", "", `${Math.round(item.difficulty * 100)}% difficulty`),
    textElement("span", "", `${item.lexical_token_count.toLocaleString()} words`),
  );
  if (item.known_share !== null) {
    const known = textElement("span", "", `≈${Math.round(item.known_share * 100)}% known`);
    known.title = "Predicted share of running words you currently know";
    meta.append(known);
  }
  if (item.status === "read" && item.last_completed_at) {
    meta.append(textElement("span", "", `Read ${formatLibraryDate(item.last_completed_at)}`));
  } else if (item.status === "in_progress" && item.opened_at) {
    meta.append(textElement("span", "", `Opened ${formatLibraryDate(item.opened_at)}`));
  } else if (item.status === "skipped" && item.skipped_at) {
    meta.append(textElement("span", "", `Skipped ${formatLibraryDate(item.skipped_at)}`));
  }
  if (item.completion_count > 1) {
    meta.append(textElement("span", "", `${item.completion_count} completed sessions`));
  }
  body.append(meta);

  const actions = document.createElement("div");
  actions.className = "text-card-actions";
  const preview = document.createElement("a");
  preview.className = "button button-quiet text-card-action";
  preview.href = textActionHref(profileId, item.id, "preview");
  preview.textContent = "Peek";
  preview.title = "Open with no learning evidence recorded";
  if (item.status === "skipped") {
    const read = textDispositionButton(item, "restore", "Read now", true);
    read.classList.add("button-primary", "text-card-primary");
    read.classList.remove("button-quiet");
    const restore = textDispositionButton(item, "restore", "Return to queue");
    actions.append(read, restore, preview);
  } else {
    const primaryAction =
      item.status === "read" ? "reread" : item.status === "in_progress" ? "continue" : "read";
    const primary = document.createElement("a");
    primary.className = "button button-primary text-card-action text-card-primary";
    primary.href = textActionHref(profileId, item.id, primaryAction);
    primary.textContent =
      item.status === "read"
        ? "Read again"
        : item.status === "in_progress"
          ? "Continue"
          : "Read now";
    actions.append(primary, preview);
    if (item.status === "queued" || item.status === "in_progress") {
      actions.append(textDispositionButton(item, "skip", "Skip for now"));
    }
    if (item.status === "queued" && readyTexts) {
      actions.append(textQueueMoveControls(item, readyTexts));
    }
  }
  card.append(body, actions);
  return card;
}

function textQueueMoveControls(
  item: TextLibraryItem,
  readyTexts: readonly TextLibraryItem[],
): HTMLElement {
  const controls = document.createElement("div");
  controls.className = "text-queue-controls";
  controls.setAttribute("role", "group");
  controls.setAttribute("aria-label", `Queue position controls for “${item.title}”`);
  for (const direction of ["up", "down"] as const) {
    const neighborLessonId = queueNeighborLessonId(readyTexts, item.id, direction);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "button button-quiet text-queue-button";
    button.dataset.queueDirection = direction;
    button.textContent = direction === "up" ? "↑" : "↓";
    button.disabled = neighborLessonId === null;
    button.setAttribute("aria-label", `Move “${item.title}” ${direction} in the ready queue`);
    button.title =
      neighborLessonId === null
        ? direction === "up"
          ? "Already first in the ready queue"
          : "Already last in the ready queue"
        : `Move ${direction} in the ready queue`;
    if (neighborLessonId !== null) {
      button.addEventListener("click", () => {
        void changeReadyQueueOrder(item, direction, neighborLessonId, button);
      });
    }
    controls.append(button);
  }
  return controls;
}

function textDispositionButton(
  item: TextLibraryItem,
  disposition: TextDisposition,
  label: string,
  openAfterRestore = false,
): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "button button-quiet text-card-action text-card-disposition";
  button.textContent = label;
  button.setAttribute(
    "aria-label",
    disposition === "skip"
      ? `Skip “${item.title}” for now`
      : openAfterRestore
        ? `Return “${item.title}” to the queue and read it now`
        : `Return “${item.title}” to the reading queue`,
  );
  button.addEventListener("click", () => {
    void changeLibraryTextDisposition(item, disposition, button, openAfterRestore);
  });
  return button;
}

async function postTextQueueMove(
  selectedLessonId: number,
  direction: TextQueueMoveDirection,
  neighborLessonId: number,
): Promise<Response> {
  return postJson(
    textQueueMovePath(profileApi, selectedLessonId),
    textQueueMovePayload(crypto.randomUUID(), direction, neighborLessonId),
  );
}

function focusQueueMoveControl(
  selectedLessonId: number,
  preferredDirection: TextQueueMoveDirection,
): void {
  const card = document.querySelector<HTMLElement>(
    `.text-card[data-lesson-id="${selectedLessonId}"]`,
  );
  if (!card) return;
  const fallbackDirection = preferredDirection === "up" ? "down" : "up";
  const target =
    card.querySelector<HTMLButtonElement>(
      `.text-queue-button[data-queue-direction="${preferredDirection}"]:not(:disabled)`,
    ) ??
    card.querySelector<HTMLButtonElement>(
      `.text-queue-button[data-queue-direction="${fallbackDirection}"]:not(:disabled)`,
    ) ??
    card.querySelector<HTMLElement>(".text-card-primary, .text-card-action");
  target?.focus({ preventScroll: true });
}

function restoreTextCardAction(
  button: HTMLButtonElement,
  card: HTMLElement | null,
  label?: string,
): void {
  setTextCardActionPending(button, false, label, card);
  button.focus({ preventScroll: true });
}

async function refreshTextsAndFocus(focusAfterRefresh: () => void): Promise<boolean> {
  try {
    await loadTexts(false);
    focusAfterRefresh();
    return true;
  } catch (error) {
    console.error(error);
    return false;
  }
}

async function changeReadyQueueOrder(
  item: TextLibraryItem,
  direction: TextQueueMoveDirection,
  neighborLessonId: number,
  button: HTMLButtonElement,
): Promise<void> {
  const card = setTextCardActionPending(button, true);
  const refresh = () => refreshTextsAndFocus(() => focusQueueMoveControl(item.id, direction));

  let response: Response;
  try {
    response = await postTextQueueMove(item.id, direction, neighborLessonId);
  } catch (error) {
    if (await refresh()) {
      showActionStatus(
        `Couldn’t confirm the move for “${item.title}”. The current queue has been reloaded.`,
      );
    } else {
      restoreTextCardAction(button, card);
      showActionStatus("The queue could not be reached. Nothing shown here has changed.");
    }
    console.error(error);
    return;
  }

  if (!response.ok) {
    const refreshed = await refresh();
    if (!refreshed) restoreTextCardAction(button, card);
    showActionStatus(
      response.status === 409
        ? refreshed
          ? "The queue changed before that move. It has been refreshed; please try again."
          : "The queue changed, but it could not be refreshed. Reload before trying again."
        : refreshed
          ? `Couldn’t move “${item.title}”. The current queue has been reloaded.`
          : `Couldn’t move “${item.title}”. Reload the library before trying again.`,
    );
    console.error(`Queue move returned ${response.status}`);
    return;
  }

  if (await refresh()) {
    showActionStatus(
      `Moved “${item.title}” ${direction === "up" ? "up" : "down"} in the ready queue.`,
    );
    return;
  }
  restoreTextCardAction(button, card);
  showActionStatus("The move was saved, but the library could not refresh. Reload to see it.");
}

async function postTextDisposition(
  selectedLessonId: number,
  disposition: TextDisposition,
): Promise<void> {
  const response = await postJson(
    textDispositionPath(profileApi, selectedLessonId, disposition),
    textDispositionPayload(crypto.randomUUID()),
  );
  if (!response.ok) {
    const label = disposition === "skip" ? "Skip" : "Restore";
    throw new Error(`${label} returned ${response.status}`);
  }
}

function focusLibraryTextCard(selectedLessonId: number): void {
  document
    .querySelector<HTMLElement>(
      `.text-card[data-lesson-id="${selectedLessonId}"] a, ` +
        `.text-card[data-lesson-id="${selectedLessonId}"] button`,
    )
    ?.focus();
}

async function changeLibraryTextDisposition(
  item: TextLibraryItem,
  disposition: TextDisposition,
  button: HTMLButtonElement,
  openAfterRestore: boolean,
): Promise<void> {
  const originalLabel = button.textContent ?? "";
  const pendingLabel = disposition === "skip" ? "Skipping…" : "Returning…";
  const card = setTextCardActionPending(button, true, pendingLabel);
  try {
    await postTextDisposition(item.id, disposition);
  } catch (error) {
    restoreTextCardAction(button, card, originalLabel);
    const action = disposition === "skip" ? "skip" : "return";
    showActionStatus(
      `Couldn’t ${action} “${item.title}”. The text has not moved; please try again.`,
    );
    console.error(error);
    return;
  }

  if (openAfterRestore) {
    showActionStatus(`Returned “${item.title}” to the queue. Opening it now…`);
    window.location.assign(textActionHref(profileId, item.id, "read"));
    return;
  }

  showActionStatus(
    disposition === "skip"
      ? `Skipped “${item.title}” for now. It remains available under Skipped.`
      : `Returned “${item.title}” to the reading queue.`,
  );
  if (await refreshTextsAndFocus(() => focusLibraryTextCard(item.id))) return;
  restoreTextCardAction(button, card, originalLabel);
  showActionStatus("The change was saved, but the library could not refresh. Reload to see it.");
}

const requestStateLabels: Record<TextRequestItem["state"], string> = {
  pending: "Queued",
  running: "Writing",
  completed: "Ready",
  failed: "Failed",
};

function renderTextRequests(requests: readonly TextRequestItem[]): void {
  const list = byId("topic-request-list");
  const recent = [...requests]
    .filter((request) => request.state === "completed" || request.state === "failed")
    .sort((left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at))
    .slice(0, 5);
  if (recent.length === 0) {
    list.replaceChildren();
    return;
  }
  const heading = textElement("p", "topic-request-list-heading", "Recent requests");
  const rows = recent.map((request) => {
    const row = document.createElement("div");
    row.className = `topic-request-item topic-request-item-${request.state}`;
    const copy = document.createElement("div");
    copy.className = "topic-request-item-copy";
    copy.append(
      textElement("strong", "", request.topic),
      textElement("span", "", request.error ?? requestStateLabels[request.state]),
    );
    row.append(copy, textElement("span", "topic-request-state", requestStateLabels[request.state]));
    if (request.lesson_id !== null) {
      const open = document.createElement("a");
      open.className = "topic-request-open";
      open.href = textActionHref(profileId, request.lesson_id, "read");
      open.textContent = "Open text";
      row.append(open);
    }
    return row;
  });
  list.replaceChildren(heading, ...rows);
}

function clearTextsPoll(): void {
  if (textsPollTimer === null) return;
  window.clearTimeout(textsPollTimer);
  textsPollTimer = null;
}

function scheduleTextsPoll(
  preparations: readonly TextPreparationItem[],
  requests: readonly TextRequestItem[],
): void {
  clearTextsPoll();
  const hasActiveRequest = requests.some(
    (request) => request.state === "pending" || request.state === "running",
  );
  if (preparations.length === 0 && !hasActiveRequest) {
    return;
  }
  textsPollTimer = window.setTimeout(() => {
    textsPollTimer = null;
    void loadTexts(false).catch(showError);
  }, 5_000);
}

function renderTtsSettings(settings: TtsSettings): void {
  ttsVoiceSelect.replaceChildren();
  if (!settings.supported || settings.voices.length === 0) {
    const option = document.createElement("option");
    option.textContent = `Local neural speech is not available for ${settings.learning_language}`;
    ttsVoiceSelect.append(option);
    ttsVoiceSelect.disabled = true;
  } else {
    for (const voice of settings.voices) {
      const option = document.createElement("option");
      option.value = voice.id;
      option.textContent = `${voice.label} — ${voice.description} · native: ${voice.native_language}${
        voice.recommended ? " · recommended" : ""
      }`;
      option.selected = voice.id === settings.selected_voice_id;
      ttsVoiceSelect.append(option);
    }
    ttsVoiceSelect.disabled = false;
  }
  byId("tts-language-note").textContent =
    settings.note ??
    (settings.supported
      ? `Qwen uses its ${settings.model_language ?? "multilingual"} mode for this profile.`
      : "Listen will use a compatible browser or operating-system voice when one exists.");
  ttsSettingsStatus.textContent = settings.provider_installed
    ? "Qwen3-TTS is installed. Missing audio is prepared one text at a time."
    : "The optional local Qwen runtime is not installed; Listen will use a system voice.";
}

function renderReadingPreferences(state: ReadingPreferences): void {
  preferencesRevision = state.revision_id;
  preferencesText.value = state.text;
  preferencesText.disabled = false;
  preferencesSave.disabled = false;
  const updated = state.updated_at
    ? `${state.source === "agent" ? "Updated by the agent" : "Saved by you"} ${formatLibraryDate(
        state.updated_at,
      )}`
    : "Starting notes from your profile interests";
  byId("preferences-meta").textContent = state.last_agent_reason
    ? `${updated}. Last agent change: ${state.last_agent_reason}`
    : updated;
  const pending = state.pending_messages.length;
  byId("preferences-pending").textContent = pending
    ? `${pending} message${pending === 1 ? "" : "s"} waiting for the agent`
    : "";
}

async function loadReadingPreferences(): Promise<void> {
  renderReadingPreferences(
    parseReadingPreferences(await getJson(`${profileApi}/reading-preferences`, "Preferences")),
  );
}

async function saveReadingPreferences(): Promise<void> {
  preferencesSave.disabled = true;
  preferencesStatus.textContent = "Saving notes…";
  try {
    const response = await postJson(
      `${profileApi}/reading-preferences`,
      { text: preferencesText.value, expected_revision_id: preferencesRevision },
      "PUT",
    );
    if (response.status === 409) {
      throw new Error("The notes changed meanwhile (the agent may have updated them). Reload.");
    }
    if (!response.ok) throw new Error(`Saving notes returned ${response.status}`);
    renderReadingPreferences(parseReadingPreferences(await response.json()));
    preferencesStatus.textContent = "Notes saved. New texts will follow them.";
  } catch (error) {
    preferencesSave.disabled = false;
    preferencesStatus.textContent =
      error instanceof Error ? error.message : "The notes could not be saved.";
  }
}

async function sendPreferenceMessage(): Promise<void> {
  const message = preferencesMessage.value.trim();
  if (!message) return;
  preferencesSend.disabled = true;
  preferencesStatus.textContent = "Sending…";
  try {
    const response = await postJson(`${profileApi}/reading-preferences/messages`, {
      message_id: crypto.randomUUID(),
      text: message,
    });
    if (!response.ok) throw new Error(`Sending returned ${response.status}`);
    renderReadingPreferences(parseReadingPreferences(await response.json()));
    preferencesMessage.value = "";
    preferencesStatus.textContent = "Sent. The agent folds it into the notes within a few minutes.";
  } catch (error) {
    preferencesStatus.textContent =
      error instanceof Error ? error.message : "The message could not be sent.";
  } finally {
    preferencesSend.disabled = false;
  }
}

async function loadTtsSettings(): Promise<void> {
  renderTtsSettings(parseTtsSettings(await getJson(`${profileApi}/tts`, "Voice settings")));
}

async function updateTtsVoice(voiceId: string): Promise<void> {
  ttsVoiceSelect.disabled = true;
  ttsSettingsStatus.textContent = "Saving voice and scheduling your text library…";
  try {
    const response = await postJson(`${profileApi}/tts`, { voice_id: voiceId }, "PUT");
    if (!response.ok) throw new Error(`Voice update returned ${response.status}`);
    renderTtsSettings(parseTtsSettings(await response.json()));
    ttsSettingsStatus.textContent =
      "Voice saved. New and existing texts will use it as their audio becomes ready.";
  } catch (error) {
    ttsVoiceSelect.disabled = false;
    ttsSettingsStatus.textContent =
      error instanceof Error ? error.message : "The voice could not be saved.";
  }
}

function renderWords(state: WordsPayload): void {
  wordsPayload = state;
  configureToneColors(state.learning_language);
  const totals = state.words.reduce(
    (result, word) => ({
      appearances: result.appearances + word.occurrence_count,
      clicks: result.clicks + word.raw_reveal_count,
      counted: result.counted + word.counted_reveal_sessions,
      cleanReads: result.cleanReads + word.qualified_exposures,
    }),
    { appearances: 0, clicks: 0, counted: 0, cleanReads: 0 },
  );
  const summary = byId("words-summary");
  summary.replaceChildren(
    summaryStat("words-summary-stat", state.words.length, "words"),
    summaryStat("words-summary-stat", totals.appearances, "appearances"),
    summaryStat("words-summary-stat", totals.clicks, "popup clicks"),
    summaryStat("words-summary-stat", totals.counted, "SRS-attributed checks"),
    summaryStat("words-summary-stat", totals.cleanReads, "clean reads"),
  );
  renderWordTable();
  setView("words");
}

function renderCharacters(state: CharactersPayload): void {
  charactersPayload = state;
  configureToneColors(state.learning_language);
  const summary = summarizeCharacters(state.characters);
  byId("characters-summary").replaceChildren(
    summaryStat("characters-summary-stat", summary.unique_characters, "characters"),
    summaryStat("characters-summary-stat", summary.appearances, "appearances"),
    summaryStat("characters-summary-stat", summary.qualified_exposures, "clean reading signals"),
    summaryStat(
      "characters-summary-stat",
      summary.inferred_failure_mass,
      "weighted difficulty signals",
    ),
  );
  renderCharacterTable();
  setView("characters");
}

function renderStatistics(state: StatisticsPayload): void {
  configureToneColors(state.learning_language);
  renderStatisticsView(state);
  setView("statistics");
}

function renderGrammar(state: GrammarState): void {
  grammarPayload = state;
  configureToneColors(state.learning_language);
  const statuses = state.constructions.map(constructionStatus);
  const directHelp = state.constructions.reduce(
    (total, construction) => total + construction.counted_help_sessions,
    0,
  );
  const inferredHelp = state.constructions.reduce(
    (total, construction) => total + construction.inferred_difficulty_signals,
    0,
  );
  byId("grammar-summary").replaceChildren(
    summaryStat("grammar-summary-stat", state.constructions.length, "constructions"),
    summaryStat(
      "grammar-summary-stat",
      statuses.filter((status) => status === "needs_attention").length,
      "need attention",
    ),
    summaryStat(
      "grammar-summary-stat",
      statuses.filter((status) => status === "comfortable").length,
      "comfortable",
    ),
    summaryStat("grammar-summary-stat", directHelp + inferredHelp, "help signals"),
  );
  renderGrammarCards();
  setView("grammar");
}

function renderGrammarCards(): void {
  if (!grammarPayload) return;
  const selected = selectGrammar(
    grammarPayload.constructions,
    byId<HTMLInputElement>("grammar-search").value,
    grammarSort,
  );
  byId<HTMLSelectElement>("grammar-sort").value = grammarSort;
  byId("grammar-card-list").replaceChildren(...selected.map(renderGrammarCard));
  byId("grammar-empty").hidden = selected.length > 0;
}

function renderGrammarCard(construction: GrammarConstruction): HTMLElement {
  const card = document.createElement("article");
  const status = constructionStatus(construction);
  card.className = `grammar-card grammar-card-${status.replace("_", "-")}`;

  const heading = document.createElement("header");
  heading.className = "grammar-card-heading";
  const labels = document.createElement("div");
  labels.className = "grammar-card-labels";
  const statusLabel = textElement(
    "span",
    `grammar-status grammar-status-${status.replace("_", "-")}`,
    formatGrammarStatus(construction),
  );
  statusLabel.title = CONSTRUCTION_STATUS_DETAILS[status].description;
  labels.append(
    statusLabel,
    textElement("span", "grammar-category", construction.category),
    textElement(
      "span",
      "grammar-difficulty",
      `${Math.round(construction.difficulty * 100)}% level`,
    ),
  );
  heading.append(
    labels,
    textElement("h2", "grammar-card-title", construction.label),
    textElement("p", "grammar-card-description", construction.description),
  );

  const metrics = document.createElement("dl");
  metrics.className = "grammar-card-metrics";
  metrics.append(
    grammarMetric(
      "Met",
      construction.occurrence_count,
      `${construction.exposed_lesson_count} ${plural(construction.exposed_lesson_count, "text")}`,
    ),
    grammarMetric(
      "Help",
      construction.counted_help_sessions,
      construction.inferred_difficulty_signals > 0
        ? `+${construction.inferred_difficulty_signals} weak sentence ${plural(
            construction.inferred_difficulty_signals,
            "signal",
          )}`
        : "No sentence signals",
    ),
    grammarMetric(
      "Clean reads",
      construction.qualified_exposures,
      formatGrammarDue(construction.next_due_at),
    ),
    grammarMetric("Confidence", `${Math.round(construction.mastery * 100)}%`, "Estimate"),
  );

  const evidence = document.createElement("details");
  evidence.className = "grammar-evidence";
  const evidenceSummary = document.createElement("summary");
  evidenceSummary.textContent = "How this estimate was formed";
  const evidenceCopy = document.createElement("p");
  evidenceCopy.textContent = `${construction.counted_help_sessions} direct-help ${plural(
    construction.counted_help_sessions,
    "session",
  )}, ${construction.inferred_difficulty_signals} weak sentence-help ${plural(
    construction.inferred_difficulty_signals,
    "signal",
  )}, and ${construction.qualified_exposures} sufficiently read completion ${plural(
    construction.qualified_exposures,
    "signal",
  )}. Reopening the same explanation in one session does not add another penalty.`;
  evidence.append(evidenceSummary, evidenceCopy);
  if (construction.raw_help_count > construction.counted_help_sessions) {
    evidence.append(
      textElement(
        "p",
        "grammar-evidence-raw",
        `${construction.raw_help_count.toLocaleString()} raw explanation opens are retained for inspection.`,
      ),
    );
  }

  card.append(heading, metrics, evidence);
  if (construction.examples.length > 0) {
    const examples = document.createElement("section");
    examples.className = "grammar-examples";
    examples.append(textElement("h3", "", "From your texts"));
    const [first, ...rest] = construction.examples;
    if (first) examples.append(renderGrammarExample(first));
    if (rest.length > 0) {
      const more = document.createElement("details");
      more.className = "grammar-more-examples";
      const summary = document.createElement("summary");
      summary.textContent = `${rest.length} more ${plural(rest.length, "example")}`;
      more.append(summary, ...rest.map(renderGrammarExample));
      examples.append(more);
    }
    card.append(examples);
  }
  return card;
}

function grammarMetric(label: string, value: number | string, detail: string): HTMLElement {
  const item = document.createElement("div");
  item.append(
    textElement("dt", "", label),
    textElement("dd", "grammar-metric-value", String(value)),
    textElement("dd", "grammar-metric-detail", detail),
  );
  return item;
}

function renderGrammarExample(example: GrammarConstruction["examples"][number]): HTMLElement {
  const exampleCard = document.createElement("article");
  exampleCard.className = "grammar-example";
  const context = textElement("p", "grammar-example-source", example.text);
  context.dir = "auto";
  exampleCard.append(context);
  if (example.translation) {
    const translation = textElement("p", "grammar-example-translation", example.translation);
    translation.dir = "auto";
    exampleCard.append(translation);
  }
  if (example.note) exampleCard.append(textElement("p", "grammar-example-note", example.note));
  const footer = document.createElement("footer");
  footer.append(textElement("span", "grammar-example-title", example.title));
  const link = document.createElement("a");
  link.className = "grammar-example-link";
  link.href = textActionHref(profileId, example.lesson_id, "preview");
  link.textContent = "Peek at text";
  link.title = "Open this text without recording learning evidence";
  footer.append(link);
  exampleCard.append(footer);
  return exampleCard;
}

function renderCharacterTable(): void {
  const state = charactersPayload;
  if (!state) return;
  const selected = selectCharacters(
    state.characters,
    byId<HTMLInputElement>("characters-search").value,
    characterSortState.sort,
  );
  if (reverseCharacterSortResult(characterSortState)) selected.reverse();
  updateCharacterSortControls();
  byId<HTMLTableSectionElement>("characters-table-body").replaceChildren(
    ...selected.map((entry) => renderCharacterRow(entry, state.learning_language)),
  );
  byId("characters-empty").hidden = selected.length > 0;
}

interface SortControlsConfig<S extends string> {
  selectId: string;
  buttonSelector: string;
  datasetKey: "wordSort" | "characterSort";
  sorts: readonly S[];
}

const WORD_SORT_CONTROLS: SortControlsConfig<WordSort> = {
  selectId: "words-sort",
  buttonSelector: "[data-word-sort]",
  datasetKey: "wordSort",
  sorts: WORD_SORTS,
};

const CHARACTER_SORT_CONTROLS: SortControlsConfig<CharacterSort> = {
  selectId: "characters-sort",
  buttonSelector: "[data-character-sort]",
  datasetKey: "characterSort",
  sorts: CHARACTER_SORTS,
};

function updateSortControls<S extends string>(
  controls: SortControlsConfig<S>,
  state: { sort: S; direction: "ascending" | "descending" },
  selection: string,
): void {
  byId<HTMLSelectElement>(controls.selectId).value = selection;
  for (const button of document.querySelectorAll<HTMLButtonElement>(controls.buttonSelector)) {
    const sortValue = button.dataset[controls.datasetKey];
    if (!sortValue || !controls.sorts.includes(sortValue as S)) continue;
    const header = button.closest("th");
    if (!header) continue;
    const label = button.dataset.sortLabel ?? button.textContent?.trim() ?? "Column";
    const indicator = button.querySelector<HTMLElement>(".words-sort-indicator");
    const active = state.sort === sortValue;
    if (!active) {
      header.removeAttribute("aria-sort");
      button.setAttribute("aria-label", `Sort by ${label}`);
      button.title = `Sort by ${label}`;
      if (indicator) indicator.textContent = "↕";
      continue;
    }
    header.setAttribute("aria-sort", state.direction);
    const nextDirection = state.direction === "ascending" ? "descending" : "ascending";
    button.setAttribute("aria-label", `${label}, sorted ${state.direction}. Sort ${nextDirection}`);
    button.title = `Sort ${label} ${nextDirection}`;
    if (indicator) indicator.textContent = state.direction === "ascending" ? "↑" : "↓";
  }
}

function updateCharacterSortControls(): void {
  updateSortControls(
    CHARACTER_SORT_CONTROLS,
    characterSortState,
    characterSortSelection(characterSortState),
  );
}

function renderCharacterRow(entry: CharacterView, learningLanguage: string): HTMLTableRowElement {
  const row = document.createElement("tr");
  const identity = wordCell("Character");
  const glyph = textElement("p", "character-glyph", entry.character);
  glyph.lang = learningLanguage;
  identity.append(
    glyph,
    textElement("p", "character-date", `Last met ${formatLibraryDate(entry.last_exposed_at)}`),
  );

  const met = metricCell(
    "Met",
    entry.occurrence_count,
    `${entry.exposed_lesson_count} opened ${plural(entry.exposed_lesson_count, "text")}`,
  );
  met.title = "Literal appearances across opened titles and text";
  const contexts = metricCell(
    "Contexts",
    entry.distinct_word_contexts,
    `distinct reusable word ${plural(entry.distinct_word_contexts, "context")}`,
  );
  const checks = metricCell(
    "Checks",
    entry.raw_reveal_count,
    `${entry.counted_reveal_sessions} counted ${plural(entry.counted_reveal_sessions, "session")}`,
  );
  checks.title =
    "Popup openings for words or display spans containing this character; each check is only bounded evidence about the character.";

  const evidence = wordCell("Evidence");
  evidence.append(
    textElement("p", "word-count", entry.qualified_exposures.toLocaleString()),
    textElement(
      "p",
      "word-count-detail",
      `clean completed-reading ${plural(entry.qualified_exposures, "signal")}`,
    ),
    textElement(
      "p",
      "character-difficulty-evidence",
      entry.inferred_failure_mass > 0
        ? `${formatEvidence(entry.inferred_failure_mass)} weighted difficulty · ${
            entry.inferred_failure_sessions
          } ${plural(entry.inferred_failure_sessions, "session")}`
        : "No inferred difficulty",
    ),
  );
  if (entry.direct_successes > 0 || entry.direct_failures > 0) {
    evidence.append(
      textElement(
        "p",
        "character-direct-evidence",
        `${entry.direct_successes} direct ${plural(
          entry.direct_successes,
          "success",
        )} · ${entry.direct_failures} direct ${plural(entry.direct_failures, "failure")}`,
      ),
    );
  }

  const recognition = renderCharacterRecognition(entry);

  row.append(identity, met, contexts, checks, evidence, recognition);
  return row;
}

function renderCharacterRecognition(entry: CharacterView): HTMLTableCellElement {
  const cell = wordCell("Recognition");
  if (!entry.last_evidence_at) {
    cell.append(
      textElement("p", "character-recognition-value", "Unrated"),
      textElement("p", "word-count-detail", "No completed-reading evidence yet"),
    );
    return cell;
  }

  const currentPercent = Math.round(entry.retrievability * 100);
  const masteryPercent = Math.round(entry.mastery * 100);
  const uncertaintyPoints = Math.round(entry.mastery_uncertainty * 100);
  cell.append(textElement("p", "character-recognition-value", `${currentPercent}% now`));

  cell.append(
    masteryMeter(
      currentPercent,
      `${entry.character} estimated recognition now`,
      "character-recognition-meter",
    ),
    textElement(
      "p",
      "character-recognition-detail",
      `${masteryPercent}% at last evidence · ±${uncertaintyPoints} pts`,
    ),
    textElement(
      "p",
      "word-due",
      `${formatDue(entry.next_due_at)} · ${formatStability(entry.stability_days)}`,
    ),
  );
  cell.title = `Last evidence ${formatLibraryDate(entry.last_evidence_at)}`;
  return cell;
}

function renderWordTable(): void {
  const state = wordsPayload;
  if (!state) return;
  const selected = selectWords(
    state.words,
    byId<HTMLInputElement>("words-search").value,
    wordSortState.sort,
  );
  if (reverseWordSortResult(wordSortState)) selected.reverse();
  updateWordSortControls();
  const body = byId<HTMLTableSectionElement>("words-table-body");
  body.replaceChildren(...selected.map((word) => renderWordRow(word, state.learning_language)));
  byId("words-empty").hidden = selected.length > 0;
}

function updateWordSortControls(): void {
  updateSortControls(WORD_SORT_CONTROLS, wordSortState, wordSortSelection(wordSortState));
}

function renderWordRow(word: WordView, learningLanguage: string): HTMLTableRowElement {
  const row = document.createElement("tr");
  const identity = wordCell("Word");
  const lemma = textElement("p", "word-lemma", "");
  renderHanziTones(lemma, word.lemma, word.pronunciation ?? "");
  identity.append(lemma);
  if (word.pronunciation) {
    const pronunciation = textElement("p", "word-pronunciation", "");
    if (isChineseLanguage(learningLanguage)) {
      renderPinyin(pronunciation, word.pronunciation, word.lemma);
    } else {
      pronunciation.textContent = word.pronunciation;
    }
    identity.append(pronunciation);
  }
  if (word.frequency_rank) {
    identity.append(
      textElement("p", "word-meta", `Frequency #${word.frequency_rank.toLocaleString()}`),
    );
  }

  const meaning = wordCell("Meaning");
  const gloss = textElement("p", "word-gloss", "");
  gloss.append(textElement("span", "word-class", word.pos), document.createTextNode(word.gloss));
  meaning.append(gloss);
  if (word.related_senses.length > 0) {
    const related = document.createElement("div");
    related.className = "word-related-senses";
    related.append(textElement("p", "word-related-label", "Also in your texts"));
    const list = document.createElement("ul");
    for (const sense of word.related_senses) {
      const item = document.createElement("li");
      item.append(
        textElement("span", "word-class", sense.pos),
        document.createTextNode(sense.gloss),
      );
      if (sense.pronunciation && sense.pronunciation !== word.pronunciation) {
        const pronunciation = textElement("span", "word-related-pronunciation", "");
        if (isChineseLanguage(learningLanguage)) {
          renderPinyin(pronunciation, sense.pronunciation, word.lemma);
        } else {
          pronunciation.textContent = sense.pronunciation;
        }
        item.append(pronunciation);
      }
      list.append(item);
    }
    related.append(list);
    meaning.append(related);
  }
  const showForms =
    word.surface_forms.length > 1 ||
    word.surface_forms.some((surface) => surface.text !== word.lemma);
  if (showForms) {
    const forms = word.surface_forms
      .slice(0, 4)
      .map((surface) =>
        surface.occurrences > 1 ? `${surface.text} ×${surface.occurrences}` : surface.text,
      )
      .join(" · ");
    const remainder = word.surface_forms.length > 4 ? ` · +${word.surface_forms.length - 4}` : "";
    meaning.append(textElement("p", "word-forms", `Forms: ${forms}${remainder}`));
  }

  const met = metricCell(
    "Met",
    word.occurrence_count,
    `${word.exposed_lesson_count} ${plural(word.exposed_lesson_count, "lesson")}`,
  );
  const clicks = metricCell(
    "Clicks",
    word.raw_reveal_count,
    `${word.counted_reveal_sessions} SRS-attributed ${plural(
      word.counted_reveal_sessions,
      "check",
    )}`,
  );
  clicks.title =
    "Literal popup openings, with direct or compound-attributed lesson sessions underneath";
  const cleanReads = metricCell(
    "Clean reads",
    word.qualified_exposures,
    word.reveal_failures > 0
      ? `${formatEvidence(word.reveal_failures)} weighted reveal evidence`
      : "No counted reveals",
  );

  const mastery = wordCell("Mastery");
  const percent = Math.round(word.mastery * 100);
  mastery.append(textElement("p", "word-count", `${percent}%`));
  mastery.append(
    masteryMeter(percent, `${word.lemma} estimated mastery`),
    textElement("p", "word-due", formatDue(word.next_due_at)),
  );

  row.append(identity, meaning, met, clicks, cleanReads, mastery);
  return row;
}

function wordCell(label: string): HTMLTableCellElement {
  const cell = document.createElement("td");
  cell.dataset.label = label;
  return cell;
}

function metricCell(label: string, value: number, detail: string): HTMLTableCellElement {
  const cell = wordCell(label);
  cell.append(
    textElement("p", "word-count", value.toLocaleString()),
    textElement("p", "word-count-detail", detail),
  );
  return cell;
}

function masteryMeter(percent: number, ariaLabel: string, extraClass?: string): HTMLElement {
  const meter = document.createElement("div");
  meter.className = extraClass ? `word-mastery-meter ${extraClass}` : "word-mastery-meter";
  meter.setAttribute("role", "progressbar");
  meter.setAttribute("aria-label", ariaLabel);
  meter.setAttribute("aria-valuemin", "0");
  meter.setAttribute("aria-valuemax", "100");
  meter.setAttribute("aria-valuenow", String(percent));
  const fill = document.createElement("div");
  fill.className = "word-mastery-fill";
  fill.style.width = `${percent}%`;
  meter.append(fill);
  return meter;
}

function formatEvidence(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/0+$/, "");
}

function formatStability(days: number): string {
  if (days < 1) return `${Math.max(1, Math.round(days * 24))}h stability`;
  if (days < 10) return `${days.toFixed(1).replace(/\.0$/, "")}d stability`;
  return `${Math.round(days)}d stability`;
}

function renderSentence(
  sentence: Sentence,
  previouslyRevealed: boolean,
  termBands: Readonly<Record<string, TermBand>>,
  dialogue = false,
): HTMLElement {
  const unit = document.createElement("span");
  unit.className = dialogue ? "sentence-unit sentence-dialogue" : "sentence-unit";
  for (let index = 0; index < sentence.runs.length; index += 1) {
    const run = sentence.runs[index];
    if (!run) continue;
    if (!run.term) {
      unit.append(document.createTextNode(run.text));
      continue;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "term";
    const termBand = termBands[run.term.key];
    attachTermBand(button, termBand);
    renderHanziTones(button, run.text, pronunciationForRun(run) ?? "");
    const readingCue = termBand ? `, ${termBand} reading cue` : "";
    button.setAttribute("aria-label", `${run.text}${readingCue}, show translation`);
    button.setAttribute("aria-expanded", "false");
    button.addEventListener("click", () => revealTerm(button, sentence, run));

    const nextRun = sentence.runs[index + 1];
    if (!nextRun?.term) {
      const punctuation = splitClosingPunctuation(nextRun?.text ?? "");
      if (punctuation.attached) {
        const cluster = document.createElement("span");
        cluster.className = "punctuation-cluster";
        cluster.append(button, document.createTextNode(punctuation.attached));
        unit.append(cluster);
        if (punctuation.remainder) {
          unit.append(document.createTextNode(punctuation.remainder));
        }
        index += 1;
        continue;
      }
    }
    unit.append(button);
  }

  const grammarOccurrences = visibleGrammarOccurrences(
    sentence.grammar,
    comfortableGrammarConstructionKeys,
  );
  if (grammarOccurrences.length > 0) {
    const grammarButton = document.createElement("button");
    grammarButton.type = "button";
    grammarButton.className = "grammar-reveal";
    grammarButton.dataset.readerCopyIgnore = "true";
    grammarButton.setAttribute("aria-controls", "grammar-popover");
    grammarButton.setAttribute("aria-expanded", "false");
    const count = grammarOccurrences.length;
    grammarButton.setAttribute(
      "aria-label",
      `${count} grammar ${plural(count, "note")} in this sentence`,
    );
    grammarButton.title = count === 1 ? "Grammar note" : `${count} grammar notes`;
    grammarButton.append(textElement("span", "grammar-reveal-mark", "G"));
    if (count > 1) {
      grammarButton.append(textElement("span", "grammar-reveal-count", String(count)));
    }
    if (grammarOccurrences.some((occurrence) => revealedGrammarOccurrences.has(occurrence.key))) {
      grammarButton.dataset.revealed = "true";
    }
    grammarButton.addEventListener("click", (event) => {
      event.stopPropagation();
      if (grammarHelpAnchor === grammarButton && !grammarPopover.hidden) {
        closeGrammarHelp();
        return;
      }
      openGrammarHelp(grammarButton, sentence, grammarOccurrences);
    });
    unit.append(grammarButton);
  }

  if (!sentence.translation) {
    unit.append(sentenceSeparator());
    return unit;
  }
  const reveal = document.createElement("button");
  reveal.type = "button";
  reveal.className = "sentence-reveal";
  reveal.dataset.readerCopyIgnore = "true";
  const icon = textElement("span", "", "A·文");
  icon.setAttribute("aria-hidden", "true");
  reveal.append(icon);
  reveal.setAttribute("aria-controls", "sentence-translation-popover");
  reveal.setAttribute("aria-expanded", "false");
  setSentenceButtonState(reveal, previouslyRevealed, false);

  const recordReveal = (): void => {
    if (sentenceReveals.reveal(sentence.key) && lesson) {
      recordLearningEvent("translation.revealed", {
        sentence_key: sentence.key,
        scope: "sentence",
      });
    }
    setSentenceButtonState(reveal, true, sentenceHelpAnchor === reveal);
  };

  const toggle = (): void => {
    setSentenceSelectionMode(false);
    positionSentenceControl(reveal, unit);
    if (sentenceHelpAnchor === reveal && !sentencePopover.hidden) {
      closeSentenceHelp();
      return;
    }
    openSentenceHelp(
      reveal,
      unit,
      sentence.runs.map((run) => run.text).join(""),
      sentence.translation ?? "",
    );
    if (!coarsePointer.matches) {
      recordReveal();
      return;
    }
    // Only count touch help that stays open: a quick open-and-close was a mistap.
    setSentenceButtonState(reveal, sentenceReveals.has(sentence.key), true);
    pendingSentenceEvidence = window.setTimeout(() => {
      pendingSentenceEvidence = null;
      if (sentenceHelpAnchor === reveal && !sentencePopover.hidden) recordReveal();
    }, TOUCH_SENTENCE_HELP_GRACE_MS);
  };

  let pressTimer: number | null = null;
  let pressOrigin: { x: number; y: number } | null = null;
  let suppressNextClick = false;
  const cancelPress = (): void => {
    if (pressTimer !== null) window.clearTimeout(pressTimer);
    pressTimer = null;
    pressOrigin = null;
  };
  unit.addEventListener(
    "pointerdown",
    (event) => {
      suppressNextClick = false;
      if (event.pointerType !== "touch" || sentenceSelectionMode) return;
      const target = event.target;
      if (target instanceof Element && target.closest(".sentence-reveal, .grammar-reveal")) return;
      cancelPress();
      pressOrigin = { x: event.clientX, y: event.clientY };
      pressTimer = window.setTimeout(() => {
        pressTimer = null;
        pressOrigin = null;
        suppressNextClick = true;
        window.getSelection()?.removeAllRanges();
        toggle();
      }, SENTENCE_LONG_PRESS_MS);
    },
    { passive: true },
  );
  unit.addEventListener(
    "pointermove",
    (event) => {
      if (
        pressOrigin &&
        Math.hypot(event.clientX - pressOrigin.x, event.clientY - pressOrigin.y) >
          LONG_PRESS_MOVE_TOLERANCE_PX
      ) {
        cancelPress();
      }
    },
    { passive: true },
  );
  unit.addEventListener("pointerup", cancelPress, { passive: true });
  unit.addEventListener("pointercancel", cancelPress, { passive: true });
  unit.addEventListener("contextmenu", (event) => {
    if (suppressNextClick || pressTimer !== null) event.preventDefault();
  });

  reveal.addEventListener("click", (event) => {
    event.stopPropagation();
    toggle();
  });
  reveal.addEventListener("focus", () => positionSentenceControl(reveal, unit));
  unit.addEventListener("pointermove", (event) => {
    positionSentenceControl(reveal, unit, event.clientY);
  });
  unit.addEventListener(
    "click",
    (event) => {
      if (suppressNextClick) {
        // The finger lifted after a long press; that release must not also reveal a word.
        suppressNextClick = false;
        event.preventDefault();
        event.stopPropagation();
        return;
      }
      const target = event.target;
      const targetIsTerm = target instanceof Element && Boolean(target.closest(".term"));
      const targetIsControl =
        target instanceof Element && Boolean(target.closest(".sentence-reveal, .grammar-reveal"));
      if (
        !shouldHandleSentenceClick(
          sentenceSelectionMode,
          targetIsTerm,
          targetIsControl,
          coarsePointer.matches,
        )
      ) {
        return;
      }
      const selection = window.getSelection();
      if (selection && !selection.isCollapsed) return;
      event.preventDefault();
      event.stopPropagation();
      toggle();
    },
    { capture: true },
  );
  unit.append(reveal, sentenceSeparator());
  return unit;
}

function sentenceSeparator(): HTMLElement {
  const separator = document.createElement("span");
  separator.className = "sentence-separator";
  separator.dataset.readerCopyIgnore = "true";
  separator.setAttribute("aria-hidden", "true");
  separator.textContent = " ";
  return separator;
}

function setSentenceSelectionMode(enabled: boolean): void {
  sentenceSelectionMode = enabled;
  sentenceHelpModeButton.setAttribute("aria-pressed", String(enabled));
  sentenceHelpModeButton.setAttribute(
    "aria-label",
    enabled ? "Cancel sentence translation selection" : "Choose a sentence to translate",
  );
  byId("sentence-help-mode-label").textContent = enabled ? "Tap a sentence" : "Sentence";
}

function setSentenceButtonState(
  button: HTMLButtonElement,
  revealed: boolean,
  expanded: boolean,
): void {
  button.dataset.revealed = String(revealed);
  button.setAttribute("aria-expanded", String(expanded));
  const label = expanded
    ? "Hide sentence translation"
    : revealed
      ? "Review sentence translation"
      : "Show sentence translation";
  button.setAttribute("aria-label", label);
  button.title = label;
}

function closestSentenceFragment(fragments: readonly DOMRect[], pointerY: number): number {
  let closest = 0;
  let closestDistance = Number.POSITIVE_INFINITY;
  for (const [index, fragment] of fragments.entries()) {
    const distance =
      pointerY < fragment.top
        ? fragment.top - pointerY
        : pointerY > fragment.bottom
          ? pointerY - fragment.bottom
          : 0;
    if (distance < closestDistance) {
      closest = index;
      closestDistance = distance;
    }
  }
  return closest;
}

function positionSentenceControl(
  button: HTMLButtonElement,
  unit: HTMLElement,
  pointerY?: number,
): void {
  const fragments = Array.from(unit.getClientRects()).filter((rect) => rect.width > 0);
  const storedIndex = Number(button.dataset.fragmentIndex ?? 0);
  const fallbackIndex =
    Number.isInteger(storedIndex) && storedIndex >= 0 && storedIndex < fragments.length
      ? storedIndex
      : 0;
  const fragmentIndex =
    pointerY === undefined ? fallbackIndex : closestSentenceFragment(fragments, pointerY);
  const fragment = fragments[fragmentIndex];
  if (!fragment) return;
  button.dataset.fragmentIndex = String(fragmentIndex);
  const content = lessonContent.getBoundingClientRect();
  const size = 28;
  let left = content.left - size;
  if (left < 4) left = Math.min(window.innerWidth - size - 4, content.right - size);
  button.style.setProperty("--sentence-help-left", `${Math.round(left)}px`);
  button.style.setProperty("--sentence-help-top", `${Math.round(fragment.top + 4)}px`);
}

function openSentenceHelp(
  anchor: HTMLButtonElement,
  unit: HTMLElement,
  source: string,
  translation: string,
): void {
  closeGloss();
  closeGrammarHelp();
  if (sentenceHelpAnchor && sentenceHelpAnchor !== anchor) {
    setSentenceButtonState(
      sentenceHelpAnchor,
      sentenceHelpAnchor.dataset.revealed === "true",
      false,
    );
  }
  sentenceHelpAnchor = anchor;
  sentenceHelpUnit = unit;
  sentencePopoverSource.textContent = source;
  sentencePopoverText.textContent = translation;
  sentencePopover.hidden = false;
  setSentenceButtonState(anchor, true, true);
  positionSentenceHelp();
}

function positionSentenceHelp(): void {
  if (!sentenceHelpAnchor || !sentenceHelpUnit || sentencePopover.hidden) return;
  positionSentenceControl(sentenceHelpAnchor, sentenceHelpUnit);
}

function closeSentenceHelp(returnFocus = false): void {
  if (pendingSentenceEvidence !== null) {
    window.clearTimeout(pendingSentenceEvidence);
    pendingSentenceEvidence = null;
  }
  const anchor = sentenceHelpAnchor;
  if (anchor) {
    setSentenceButtonState(anchor, anchor.dataset.revealed === "true", false);
  }
  sentenceHelpAnchor = null;
  sentenceHelpUnit = null;
  sentencePopover.hidden = true;
  if (returnFocus) anchor?.focus({ preventScroll: true });
}

function revealTerm(anchor: HTMLButtonElement, sentence: Sentence, run: Run): void {
  if (!lesson || !run.term) return;
  closeSentenceHelp();
  closeGrammarHelp();
  if (glossAnchor) glossAnchor.setAttribute("aria-expanded", "false");
  glossAnchor = anchor;
  anchor.setAttribute("aria-expanded", "true");
  byId("gloss-lemma").textContent = run.term.lemma;
  byId("gloss-pos").textContent = run.term.pos;
  byId("gloss-text").textContent = run.term.gloss;
  const pronunciation = byId("gloss-pronunciation");
  const pronunciationText = pronunciationForRun(run) ?? "";
  if (isChineseLanguage(lesson.learning_language)) {
    renderPinyin(
      pronunciation,
      pronunciationText,
      run.pronunciation === undefined ? run.term.lemma : run.text,
    );
  } else {
    pronunciation.textContent = pronunciationText;
  }
  pronunciation.hidden = !pronunciationText;
  popover.hidden = false;
  positionGloss();
  recordLearningEvent("term.revealed", {
    term_key: run.term.key,
    sentence_key: sentence.key,
  });
}

function positionPopover(
  anchorElement: HTMLElement,
  popoverElement: HTMLElement,
  offset: number,
): void {
  const anchor = anchorElement.getBoundingClientRect();
  const box = popoverElement.getBoundingClientRect();
  const margin = 12;
  let left = anchor.left + anchor.width / 2 - box.width / 2;
  left = Math.max(margin, Math.min(left, window.innerWidth - box.width - margin));
  let top = anchor.bottom + offset;
  if (top + box.height > window.innerHeight - margin) top = anchor.top - box.height - offset;
  popoverElement.style.left = `${left}px`;
  popoverElement.style.top = `${Math.max(margin, top)}px`;
}

function positionGloss(): void {
  if (!glossAnchor || popover.hidden) return;
  positionPopover(glossAnchor, popover, 10);
}

function closeGloss(): void {
  glossAnchor?.setAttribute("aria-expanded", "false");
  glossAnchor = null;
  popover.hidden = true;
}

function openGrammarHelp(
  anchor: HTMLButtonElement,
  sentence: Sentence,
  occurrences: readonly Sentence["grammar"][number][],
): void {
  closeGloss();
  closeSentenceHelp();
  if (grammarHelpAnchor && grammarHelpAnchor !== anchor) {
    grammarHelpAnchor.setAttribute("aria-expanded", "false");
  }
  grammarHelpAnchor = anchor;
  anchor.setAttribute("aria-expanded", "true");
  grammarPopoverContent.replaceChildren(
    ...occurrences.map((occurrence) => renderGrammarOccurrenceHelp(sentence, occurrence)),
  );
  grammarPopover.hidden = false;
  positionGrammarHelp();
}

function renderGrammarOccurrenceHelp(
  sentence: Sentence,
  occurrence: Sentence["grammar"][number],
): HTMLElement {
  const definition = readerGrammarCatalog.get(occurrence.construction_key);
  const item = document.createElement("section");
  item.className = "grammar-help-item";
  const header = document.createElement("div");
  header.className = "grammar-help-heading";
  header.append(
    textElement("span", "grammar-help-category", definition?.category ?? "Grammar"),
    textElement("h2", "grammar-help-label", definition?.label ?? occurrence.construction_key),
  );
  const source = textElement(
    "p",
    "grammar-help-source",
    sentence.runs
      .slice(occurrence.run_start, occurrence.run_end)
      .map((run) => run.text)
      .join(""),
  );
  source.dir = "auto";

  const explanation = document.createElement("div");
  explanation.className = "grammar-help-explanation";
  explanation.hidden = true;
  explanation.append(
    textElement(
      "p",
      "grammar-help-description",
      definition?.description ?? "This construction is not in the reader catalog.",
    ),
  );
  if (occurrence.note) {
    const note = document.createElement("p");
    note.className = "grammar-help-note";
    note.append(textElement("strong", "", "Here: "), document.createTextNode(occurrence.note));
    explanation.append(note);
  }

  const alreadyRevealed = revealedGrammarOccurrences.has(occurrence.key);
  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "grammar-help-toggle";
  toggle.dataset.revealed = String(alreadyRevealed);
  toggle.setAttribute("aria-expanded", "false");
  toggle.textContent = alreadyRevealed ? "Review explanation" : "Show explanation";
  toggle.addEventListener("click", () => {
    const show = explanation.hidden;
    explanation.hidden = !show;
    toggle.setAttribute("aria-expanded", String(show));
    toggle.textContent = show
      ? "Hide explanation"
      : revealedGrammarOccurrences.has(occurrence.key)
        ? "Review explanation"
        : "Show explanation";
    if (show && !revealedGrammarOccurrences.has(occurrence.key)) {
      revealedGrammarOccurrences.add(occurrence.key);
      toggle.dataset.revealed = "true";
      grammarHelpAnchor?.setAttribute("data-revealed", "true");
      recordLearningEvent("translation.revealed", {
        scope: "grammar",
        sentence_key: sentence.key,
        occurrence_key: occurrence.key,
        construction_key: occurrence.construction_key,
      });
    }
    positionGrammarHelp();
  });

  item.append(header, source, toggle, explanation);
  return item;
}

function positionGrammarHelp(): void {
  if (!grammarHelpAnchor || grammarPopover.hidden) return;
  positionPopover(grammarHelpAnchor, grammarPopover, 9);
}

function closeGrammarHelp(returnFocus = false): void {
  const anchor = grammarHelpAnchor;
  anchor?.setAttribute("aria-expanded", "false");
  grammarHelpAnchor = null;
  grammarPopover.hidden = true;
  if (returnFocus) anchor?.focus({ preventScroll: true });
}

function toggleFullTranslation(show: boolean): void {
  if (!lesson) return;
  fullPanel.hidden = !show;
  fullButton.textContent = show ? "Hide full translation" : "Show full translation";
  fullButton.setAttribute("aria-expanded", String(show));
  if (show && !fullTranslationRevealed) {
    fullTranslationRevealed = true;
    recordLearningEvent("translation.revealed", { scope: "lesson" });
  }
  if (show) fullPanel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

const ratings = [
  [-1, "Not for me"],
  [1, "Good lesson"],
] as const;

function renderRatings(): void {
  const container = byId("rating-options");
  container.replaceChildren();
  for (const [value, label] of ratings) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "rating-button";
    button.dataset.rating = String(value);
    button.setAttribute("aria-pressed", "false");
    button.append(textElement("strong", "rating-number", value === 1 ? "+" : "−"));
    button.append(textElement("span", "rating-label", label));
    button.addEventListener("click", () => selectRating(value));
    container.append(button);
  }
}

function selectRating(value: -1 | 1): void {
  rating = value;
  for (const button of document.querySelectorAll<HTMLButtonElement>(".rating-button")) {
    button.setAttribute("aria-pressed", String(button.dataset.rating === String(value)));
  }
}

function selectedFeedback(): string[] {
  return Array.from(
    feedbackDetails.querySelectorAll<HTMLInputElement>('input[name="lesson-feedback"]:checked'),
    (input) => input.value,
  );
}

async function completeLesson(): Promise<void> {
  if (!lesson || !readerModeRecordsEvidence(readerMode)) return;
  completeButton.disabled = true;
  completeButton.textContent = "Saving…";
  recordLearningEvent("lesson.completed", {
    active_seconds: activeTimer.seconds(),
    completion_ratio: 1,
  });
  const feedback = selectedFeedback();
  if (rating !== null || feedback.length > 0) {
    recordLearningEvent("lesson.rated", { rating, feedback });
  }
  try {
    await events.flush();
  } catch (error) {
    completeButton.disabled = false;
    completeButton.textContent = "Finish & next";
    showSyncWarning(error);
    return;
  }
  await loadLesson().catch(showError);
}

function resetReaderSkipButtons(): void {
  readerView.inert = false;
  readerView.removeAttribute("aria-busy");
  for (const button of skipLessonButtons) {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    button.textContent = "Skip for now";
  }
}

async function skipCurrentLesson(): Promise<void> {
  if (
    !lesson ||
    lessonId === null ||
    readerMode !== "standard" ||
    skipLessonButtons.every((button) => button.hidden || button.disabled)
  ) {
    return;
  }
  const skippedLessonId = lessonId;
  const skippedTitle = lesson.title;
  for (const button of skipLessonButtons) {
    button.disabled = true;
    button.setAttribute("aria-busy", "true");
    button.textContent = "Skipping…";
  }
  readerView.inert = true;
  readerView.setAttribute("aria-busy", "true");
  activeTimer.pause();
  stopLessonAudio();
  closeGloss();
  closeSentenceHelp();
  closeGrammarHelp();
  setSentenceSelectionMode(false);
  toggleFullTranslation(false);

  try {
    await events.flush();
    await postTextDisposition(skippedLessonId, "skip");
  } catch (error) {
    activeTimer.resume();
    resetReaderSkipButtons();
    showActionStatus(`Couldn’t skip “${skippedTitle}”. This text is still open; please try again.`);
    console.error(error);
    return;
  }

  showActionStatus(`Skipped “${skippedTitle}” for now. Opening the next text…`);
  window.history.replaceState(null, "", appPath(`/p/${encodeURIComponent(profileId)}`));
  try {
    await loadLesson();
    if (!readerView.hidden) byId("lesson-title").focus({ preventScroll: true });
  } catch (error) {
    showError(error);
  }
}

function emptyProgress(): ReaderProgress {
  return {
    started: false,
    completed: false,
    rating: null,
    revealed_term_keys: [],
    revealed_sentence_keys: [],
    revealed_grammar_occurrence_keys: [],
    full_translation_revealed: false,
  };
}

async function loadLesson(
  showLoading = true,
  launch: ReaderLaunch = { mode: "standard" },
): Promise<void> {
  readerMode = launch.mode;
  if (showLoading) {
    byId("loading-message").textContent =
      launch.mode === "preview"
        ? "Opening a no-progress preview…"
        : launch.mode === "reread"
          ? "Starting a fresh reread…"
          : "Preparing your next lesson…";
    setView("loading");
  }
  const payload = await getJson(
    readerApiPath(profileApi, launch),
    launch.mode === "preview" ? "Preview" : "Reader",
  );
  if (launch.mode === "preview") {
    const preview = parseTextPreview(payload);
    renderLesson(
      {
        lesson_id: preview.lesson_id,
        lesson: preview.lesson,
        progress: emptyProgress(),
        term_bands: preview.term_bands,
        grammar_catalog: preview.grammar_catalog,
        comfortable_grammar_construction_keys: preview.comfortable_grammar_construction_keys,
      },
      "preview",
      true,
    );
    completeButton.textContent = "Finish & next";
    window.scrollTo({ top: 0, behavior: "smooth" });
    return;
  }
  const state = parseReader(payload);
  configureToneColors(state.profile.learning_language);
  if (!state.lesson || state.lesson_id === null) {
    showEmpty(state.generation_status);
    return;
  }
  renderLesson(
    {
      lesson_id: state.lesson_id,
      lesson: state.lesson,
      progress: state.progress,
      term_bands: state.term_bands,
      grammar_catalog: state.grammar_catalog,
      comfortable_grammar_construction_keys: state.comfortable_grammar_construction_keys,
    },
    launch.mode,
    launch.lessonId !== undefined,
  );
  completeButton.textContent = launch.mode === "reread" ? "Finish reread" : "Finish & next";
  window.scrollTo({ top: 0, behavior: "smooth" });
}

async function loadTexts(showLoading = true): Promise<void> {
  if (showLoading) {
    byId("loading-message").textContent = "Opening your text library…";
    setView("loading");
  }
  renderTexts(parseTextLibrary(await getJson(`${profileApi}/texts`, "Texts")));
}

async function submitTopicRequest(): Promise<void> {
  const input = byId<HTMLInputElement>("topic-request-input");
  const submit = byId<HTMLButtonElement>("topic-request-submit");
  const status = byId("topic-request-status");
  const topic = input.value.trim();
  if (!topic) {
    input.setCustomValidity("Enter a topic or scenario.");
    input.reportValidity();
    return;
  }
  input.setCustomValidity("");
  submit.disabled = true;
  submit.textContent = "Requesting…";
  status.textContent = "Sending this idea to the learning agent…";
  try {
    const response = await postJson(`${profileApi}/text-requests`, {
      request_id: crypto.randomUUID(),
      topic,
    });
    if (!response.ok) throw new Error(`Topic request returned ${response.status}`);
    const request = parseTextRequest(await response.json());
    input.value = "";
    status.textContent = `Requested “${request.topic}”. Its status will update here.`;
    if (textsPayload) {
      const requests = [
        request,
        ...textsPayload.requests.filter((item) => item.request_id !== request.request_id),
      ];
      textsPayload = { ...textsPayload, requests };
      renderTextRequests(requests);
      scheduleTextsPoll(textsPayload.preparations, requests);
    }
  } catch (error) {
    status.textContent =
      error instanceof Error ? error.message : "The topic request could not be sent.";
  } finally {
    submit.disabled = false;
    submit.textContent = "Request text";
  }
}

async function loadView(
  loadingMessage: string,
  resource: string,
  failureLabel: string,
  render: (payload: unknown) => void,
): Promise<void> {
  byId("loading-message").textContent = loadingMessage;
  setView("loading");
  render(await getJson(`${profileApi}/${resource}`, failureLabel));
}

function loadWords(): Promise<void> {
  return loadView("Gathering your vocabulary…", "words", "Words", (payload) =>
    renderWords(parseWords(payload)),
  );
}

function loadCharacters(): Promise<void> {
  return loadView("Gathering your character exposure…", "characters", "Characters", (payload) =>
    renderCharacters(parseCharacters(payload)),
  );
}

function loadGrammar(): Promise<void> {
  return loadView("Gathering your grammar evidence…", "grammar", "Grammar", (payload) =>
    renderGrammar(parseGrammar(payload)),
  );
}

function loadStatistics(): Promise<void> {
  return loadView("Gathering your reading statistics…", "statistics", "Statistics", (payload) =>
    renderStatistics(parseStatistics(payload)),
  );
}

async function bootstrap(): Promise<void> {
  setView("loading");
  const activation = await loadActivation();
  if (!activation.active) {
    events.clear();
    lessonSessions.clear();
    activeTimeStore.clear();
    renderActivation();
    return;
  }
  const launch = parseReaderLaunch(window.location.search);
  readerMode = launch.mode;
  if (launch.mode === "preview") {
    await loadLesson(true, launch);
    return;
  }
  if (initialView === "settings") {
    try {
      await loadTtsSettings();
    } catch (error) {
      ttsVoiceSelect.disabled = true;
      ttsSettingsStatus.textContent =
        error instanceof Error ? error.message : "Voice settings could not be loaded.";
    }
    try {
      await loadReadingPreferences();
    } catch (error) {
      preferencesStatus.textContent =
        error instanceof Error ? error.message : "Reading preferences could not be loaded.";
    }
    setView("settings");
    void events.flush().catch(showSyncWarning);
    return;
  }
  await events.flush();
  if (initialView === "texts") await loadTexts();
  else if (initialView === "words") await loadWords();
  else if (initialView === "characters") await loadCharacters();
  else if (initialView === "grammar") await loadGrammar();
  else if (initialView === "statistics") await loadStatistics();
  else await loadLesson(true, launch);
}

function showError(error: unknown): void {
  clearLessonPoll();
  clearTextsPoll();
  const eyebrow = errorView.querySelector(".eyebrow");
  const heading = errorView.querySelector("h1");
  const viewCopy = VIEW_ERROR_COPY[initialView];
  const fallback = {
    eyebrow: readerMode === "preview" ? "Preview unavailable" : "Reader unavailable",
    heading: "We couldn’t open the lesson.",
  };
  const copy = viewCopy ?? fallback;
  if (eyebrow) eyebrow.textContent = copy.eyebrow;
  if (heading) heading.textContent = copy.heading;
  byId("retry-button").textContent = "Try again";
  byId("retry-button").hidden = false;
  byId("error-message").textContent =
    error instanceof Error ? error.message : "An unexpected error occurred.";
  setView("error");
}

function clearLessonPoll(): void {
  if (lessonPollTimer === null) return;
  window.clearTimeout(lessonPollTimer);
  lessonPollTimer = null;
}

function showEmpty(generationStatus: ReaderPayload["generation_status"]): void {
  clearLessonPoll();
  const eyebrow = errorView.querySelector(".eyebrow");
  const heading = errorView.querySelector("h1");
  const isPreparing = generationStatus === "pending" || generationStatus === "running";
  if (eyebrow) eyebrow.textContent = isPreparing ? "Lesson in progress" : "All caught up";
  if (heading) {
    heading.textContent = isPreparing
      ? "Your learning agent is preparing a new lesson."
      : "No lesson is ready yet.";
  }
  byId("error-message").textContent = isPreparing
    ? "This page will update automatically."
    : generationStatus === "failed"
      ? "The last preparation attempt failed. Try again when it has been resolved."
      : "Check again after a new lesson has been added.";
  byId("retry-button").textContent = "Check again";
  byId("retry-button").hidden = isPreparing;
  setView("error");
  if (isPreparing) {
    lessonPollTimer = window.setTimeout(() => {
      lessonPollTimer = null;
      void loadLesson(false).catch(showError);
    }, 5_000);
  }
}

function showSyncWarning(error: unknown): void {
  console.error(error);
  statusMessage.textContent = "Progress could not sync yet. We’ll retry with the next action.";
  statusMessage.hidden = false;
  window.setTimeout(() => {
    statusMessage.hidden = true;
  }, 4_000);
}

fullButton.addEventListener("click", () => toggleFullTranslation(fullPanel.hidden));
lessonAudioButton.addEventListener("click", () => void toggleLessonAudio());
lessonAudio.addEventListener("play", () => setAudioButton(true));
lessonAudio.addEventListener("pause", () => setAudioButton(false));
lessonAudio.addEventListener("ended", () => {
  lessonAudio.currentTime = 0;
  setAudioButton(false);
});
lessonAudio.addEventListener("error", () => {
  if (!lessonAudio.currentSrc) return;
  currentAudioStatus = null;
  const fallback = browserSpeechFallback(lesson?.learning_language ?? "this language");
  lessonAudioButton.disabled = !fallback.available;
  lessonAudioStatus.textContent = `${localAudioElementFailure(
    lessonAudio.error?.code ?? null,
  )} ${browserFallbackMessage(fallback)}`;
});
lessonAudioSpeed.addEventListener("change", () => {
  const rate = persistPlaybackRate(window.localStorage, profileId, lessonAudioSpeed.value);
  lessonAudioSpeed.value = String(rate);
  lessonAudio.playbackRate = rate;
});
if (browserSpeechAvailable()) {
  window.speechSynthesis.addEventListener("voiceschanged", () => {
    if (!lesson || !currentAudioStatus || currentAudioStatus.state === "ready") return;
    lessonAudioStatus.textContent = audioStatusMessage(
      currentAudioStatus,
      browserSpeechFallback(lesson.learning_language),
    );
  });
}
byId("close-translation-button").addEventListener("click", () => toggleFullTranslation(false));
byId("gloss-close").addEventListener("click", closeGloss);
byId("sentence-translation-close").addEventListener("click", () => closeSentenceHelp(true));
byId("grammar-popover-close").addEventListener("click", () => closeGrammarHelp(true));
sentenceHelpModeButton.addEventListener("click", () => {
  closeGloss();
  closeGrammarHelp();
  closeSentenceHelp();
  setSentenceSelectionMode(!sentenceSelectionMode);
});
toneColorToggle.addEventListener("click", () => {
  const enabled = toneColorToggle.getAttribute("aria-pressed") !== "true";
  toneColorPreference.set(enabled);
  configureToneColors(currentLearningLanguage);
});
for (const input of document.querySelectorAll<HTMLInputElement>('input[name="color-theme"]')) {
  input.addEventListener("change", () => {
    if (!input.checked) return;
    const theme = parseThemeMode(input.value);
    if (!theme) return;
    setTheme(persistThemePreference(window.localStorage, theme), true);
  });
}
ttsVoiceSelect.addEventListener("change", () => {
  if (ttsVoiceSelect.value) void updateTtsVoice(ttsVoiceSelect.value);
});
byId("retry-button").addEventListener("click", () => void bootstrap().catch(showError));
completeButton.addEventListener("click", () => void completeLesson());
for (const button of skipLessonButtons) {
  button.addEventListener("click", () => void skipCurrentLesson());
}
byId("topic-request-form").addEventListener("submit", (event) => {
  event.preventDefault();
  void submitTopicRequest();
});
byId("activation-form").addEventListener("submit", (event) => {
  event.preventDefault();
  void submitActivation();
});
byId("topic-request-input").addEventListener("input", (event) => {
  if (event.currentTarget instanceof HTMLInputElement) event.currentTarget.setCustomValidity("");
});
byId("reset-active-time").addEventListener("click", () => activeTimer.reset());
preferencesSave.addEventListener("click", () => void saveReadingPreferences());
preferencesSend.addEventListener("click", () => void sendPreferenceMessage());
byId("words-search").addEventListener("input", renderWordTable);
byId("characters-search").addEventListener("input", renderCharacterTable);
byId("grammar-search").addEventListener("input", renderGrammarCards);
byId("grammar-sort").addEventListener("change", (event) => {
  if (!(event.currentTarget instanceof HTMLSelectElement)) return;
  const selected = event.currentTarget.value;
  if (!GRAMMAR_SORTS.includes(selected as GrammarSort)) return;
  grammarSort = selected as GrammarSort;
  renderGrammarCards();
});
byId("words-sort").addEventListener("change", (event) => {
  if (!(event.currentTarget instanceof HTMLSelectElement)) return;
  wordSortState = parseWordSortSelection(event.currentTarget.value);
  renderWordTable();
});
byId("characters-sort").addEventListener("change", (event) => {
  if (!(event.currentTarget instanceof HTMLSelectElement)) return;
  characterSortState = parseCharacterSortSelection(event.currentTarget.value);
  renderCharacterTable();
});
function attachSortHeaderButtons<S extends string>(
  controls: SortControlsConfig<S>,
  applySort: (sort: S) => void,
): void {
  for (const button of document.querySelectorAll<HTMLButtonElement>(controls.buttonSelector)) {
    button.addEventListener("click", () => {
      const sortValue = button.dataset[controls.datasetKey];
      if (!sortValue || !controls.sorts.includes(sortValue as S)) return;
      applySort(sortValue as S);
    });
  }
}
attachSortHeaderButtons(WORD_SORT_CONTROLS, (sort) => {
  wordSortState = nextWordSort(wordSortState, sort);
  renderWordTable();
});
attachSortHeaderButtons(CHARACTER_SORT_CONTROLS, (sort) => {
  characterSortState = nextCharacterSort(characterSortState, sort);
  renderCharacterTable();
});
profileSwitch.addEventListener("change", () => {
  if (profileSwitch.value === profileId) return;
  stopLessonAudio(true);
  activeTimer.save();
  if (readerModeRecordsEvidence(readerMode)) events.sendOnExit();
  const suffix = initialView === "reading" ? "" : `?view=${initialView}`;
  window.location.assign(appPath(`/p/${encodeURIComponent(profileSwitch.value)}${suffix}`));
});
document.addEventListener("pointerdown", (event) => {
  const target = event.target;
  if (target instanceof Node && !popover.contains(target) && !glossAnchor?.contains(target)) {
    closeGloss();
  }
  if (
    target instanceof Node &&
    !sentencePopover.contains(target) &&
    !sentenceHelpUnit?.contains(target)
  ) {
    closeSentenceHelp();
  }
  if (
    target instanceof Node &&
    !grammarPopover.contains(target) &&
    !grammarHelpAnchor?.contains(target)
  ) {
    closeGrammarHelp();
  }
});
document.addEventListener("copy", (event) => {
  if (!lesson || readerView.hidden) return;
  copyReaderSelection(
    event,
    window.getSelection(),
    byId("lesson-title"),
    lessonContent,
    lesson.learning_language,
  );
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeGloss();
    closeSentenceHelp(true);
    closeGrammarHelp(true);
    setSentenceSelectionMode(false);
  }
});
window.addEventListener("resize", positionGloss);
window.addEventListener("resize", positionSentenceHelp);
window.addEventListener("resize", positionGrammarHelp);
window.addEventListener(
  "scroll",
  () => {
    positionGloss();
    positionSentenceHelp();
    positionGrammarHelp();
  },
  { passive: true },
);
window.addEventListener("pagehide", () => {
  stopLessonAudio();
  activeTimer.pagehide();
  if (readerModeRecordsEvidence(readerMode)) events.sendOnExit();
});

void bootstrap().catch(showError);
