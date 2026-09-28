"""Deterministic content blueprints that keep generated lessons structurally varied."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from server.schemas import AgentBrief, FeedbackTag, GenerationContentPlan


@dataclass(frozen=True)
class _Archetype:
    key: str
    discourse_form: str
    perspective: str
    progression: str
    ending_shape: str


_ARCHETYPES = (
    _Archetype(
        "decision-dialogue",
        "A dialogue built from short, consequential turns rather than narrated group activity.",
        "Two speakers with different practical motives; neither is simply right.",
        "Introduce a concrete choice, reveal one cost of each option, and let the speakers revise "
        "their positions before choosing.",
        "A decision with a visible trade-off, not universal agreement or celebration.",
    ),
    _Archetype(
        "mistaken-assumption",
        "A first-person personal account focused on one tightly bounded incident.",
        "One narrator whose initial interpretation turns out to be incomplete.",
        "Open with a specific surprising detail, add two clues, then revise the narrator's "
        "understanding.",
        "A changed interpretation or remaining doubt, without a moral summary.",
    ),
    _Archetype(
        "micro-mystery",
        "A compact third-person scene with a small mystery grounded in ordinary details.",
        "Stay close to one person's observations and decisions.",
        "Present an unexplained detail, test at least two plausible explanations, and discover "
        "one concrete cause.",
        "A discovery that changes the next action; it need not make anyone happy.",
    ),
    _Archetype(
        "how-it-works",
        "A concise explanatory article using one concrete example rather than a broad overview.",
        "A neutral explainer addressing a curious reader.",
        "Start from a practical question, trace cause and effect through a real example, and "
        "contrast it with one common misconception.",
        "A qualified factual takeaway followed by one open question.",
    ),
    _Archetype(
        "message-chain",
        "A short chain of messages whose meaning changes as new information arrives.",
        "Two or three distinct voices with clear immediate goals.",
        "Begin with an ambiguous request, expose a misunderstanding through replies, and clarify "
        "the concrete next step.",
        "A specific plan or unanswered reply, not a narrated happy ending.",
    ),
    _Archetype(
        "contrast-report",
        "A descriptive report comparing two objects, methods, places, or moments.",
        "An attentive observer who distinguishes fact from opinion.",
        "Establish a useful comparison, give sensory or measurable details on both sides, and "
        "show why the difference matters.",
        "A nuanced preference or contrast, not a winner-takes-all conclusion.",
    ),
    _Archetype(
        "mini-interview",
        "A brief interview organized around specific questions and concrete answers.",
        "A curious interviewer and someone with firsthand experience.",
        "Move from what happened, to why, to one drawback or unresolved difficulty.",
        "An honest limitation or next question rather than generic praise.",
    ),
    _Archetype(
        "field-notes",
        "Diary or field notes covering a short interval through precise observations.",
        "One observer who notices change without organizing a group project.",
        "Accumulate several concrete details, notice a pattern, and offer a tentative explanation.",
        "A quiet image, prediction, or question; leave some uncertainty intact.",
    ),
    _Archetype(
        "speculative-consequence",
        "A near-future vignette built around one altered rule, tool, or capability.",
        "One person directly affected by both its benefit and its cost.",
        "Demonstrate the apparent solution, reveal an unintended consequence, and require a new "
        "choice.",
        "A complication or compromise, not effortless technological success.",
    ),
    _Archetype(
        "process-interrupted",
        "A practical process shown through action, including a meaningful interruption.",
        "One learner or worker solving a specific problem without a cheering group.",
        "Follow two or three concrete steps, let one fail for an intelligible reason, and diagnose "
        "what must change.",
        "A corrected step or useful warning, not 'everyone was satisfied.'",
    ),
)

_GENERAL_SEEDS = (
    "A delayed evening train, an unclear announcement, and one passenger deciding what to do.",
    "A second-hand shop examining an unusual object whose age or purpose is uncertain.",
    "Two versions of a family recipe that disagree on one important step.",
    "A small weather station receives one measurement that does not fit the forecast.",
    "A delivery contains the correct label but the wrong object inside.",
    "An old photograph appears to show a detail that should not yet have existed.",
    "A familiar walking route is unexpectedly closed, revealing a place usually overlooked.",
    "A night-shift worker notices a machine behaving differently from the written instructions.",
    "A market seller and a customer disagree about which sign shows that food is fresh.",
    "A simple near-future device saves time but quietly creates a new inconvenience.",
)

_GENERIC_PATTERN_WARNING = (
    "Do not default to the recurring template: 'last week' or 'one day,' a group gathers to "
    "learn or improve a shared place, the atmosphere and people are praised, a token difficulty "
    "is quickly overcome, and everyone ends happy."
)


def build_content_plan(
    brief: AgentBrief,
    *,
    task_id: int,
    requested_topic: str | None,
    feedback: Sequence[FeedbackTag] = (),
) -> GenerationContentPlan:
    """Choose a coherent blueprint while avoiding recently planned archetypes."""

    recent_archetypes = _recent_plan_values(brief, "archetype", limit=4)
    archetype = _choose_unused(_ARCHETYPES, recent_archetypes, task_id)
    concrete_seed = _content_seed(
        brief,
        task_id=task_id,
        requested_topic=requested_topic,
        feedback=feedback,
    )
    recent_labels = [
        " / ".join(value for value in (lesson.title, lesson.topic) if value)
        for lesson in brief.recent_lessons[:5]
    ]
    avoid_patterns = [_GENERIC_PATTERN_WARNING]
    if recent_labels:
        avoid_patterns.append(
            "Treat these recent central situations as negative examples for repetition unless the "
            "user explicitly requested the same topic: " + "; ".join(recent_labels)
        )
    return GenerationContentPlan(
        archetype=archetype.key,
        discourse_form=archetype.discourse_form,
        perspective=archetype.perspective,
        concrete_seed=concrete_seed,
        progression=archetype.progression,
        ending_shape=archetype.ending_shape,
        avoid_patterns=avoid_patterns,
    )


def content_plan_instructions() -> str:
    return (
        "Content variety is a required quality constraint. Follow content_plan as a concrete "
        "structural blueprint, not as text to quote or mention. Treat recent_lessons as negative "
        "examples for repeated setting, social setup, progression, and ending unless explicit "
        "same-topic feedback or requested_topic says otherwise. Even then, change the discourse "
        "form, immediate goal, progression, and ending. At lower proficiency, simplify vocabulary "
        "and syntax while preserving specific objects, motives, contrasts, and consequences; do "
        "not flatten the content into a generic cooperative success story. Do not add a final "
        "sentence merely to state that everyone felt happy or learned an important lesson. Choose "
        "the situation and structure before selecting SRS terms. Inspect beyond the first target "
        "candidate and use a lower candidate or none when the highest-ranked words would distort "
        "the text."
    )


def _content_seed(
    brief: AgentBrief,
    *,
    task_id: int,
    requested_topic: str | None,
    feedback: Sequence[FeedbackTag],
) -> str:
    if requested_topic is not None:
        return (
            f"Narrow the requested topic {requested_topic!r} to one specific situation, question, "
            "or disagreement with concrete objects and a clear time and place."
        )
    if "same_topic" in feedback and brief.recent_lessons:
        recent = brief.recent_lessons[0]
        label = recent.topic or recent.title
        return (
            f"Take a genuinely different angle on the recent topic {label!r}; change the actors, "
            "immediate goal, setting details, and outcome."
        )
    if brief.profile.interests:
        interest = brief.profile.interests[task_id % len(brief.profile.interests)]
        return (
            f"Use the profile interest {interest!r}, narrowed to one specific incident, object, "
            "question, or disagreement rather than a broad survey."
        )
    return _GENERAL_SEEDS[task_id % len(_GENERAL_SEEDS)]


def _recent_plan_values(brief: AgentBrief, key: str, *, limit: int) -> set[str]:
    values: list[str] = []
    for lesson in brief.recent_lessons:
        plan = lesson.metadata.get("content_plan")
        if not isinstance(plan, dict):
            continue
        value = plan.get(key)
        if isinstance(value, str):
            values.append(value)
        if len(values) >= limit:
            break
    return set(values)


def _choose_unused(choices: Sequence[_Archetype], recent_keys: set[str], seed: int) -> _Archetype:
    for offset in range(len(choices)):
        candidate = choices[(seed + offset) % len(choices)]
        if candidate.key not in recent_keys:
            return candidate
    return choices[seed % len(choices)]
