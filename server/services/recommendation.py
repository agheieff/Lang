"""Text recommendation engine for matching texts to user profiles."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter

from sqlalchemy.orm import Session

from server.models import (
    Profile,
    ReadingText,
    ProfileTopicPref,
    ProfileTextRead,
    ProfileTextQueue,
    Lexeme,
    ReadingWordGloss,
)

logger = logging.getLogger(__name__)


# Available topics for text generation
AVAILABLE_TOPICS = [
    "fiction",
    "news",
    "science",
    "technology",
    "history",
    "daily_life",
    "culture",
    "sports",
    "business",
]


def select_topic_for_profile(db: Session, profile: Profile) -> Optional[str]:
    """Select a topic based on profile preferences with weighted random.

    Returns None if no preferences set (will randomize).
    """
    topic_prefs = (
        db.query(ProfileTopicPref)
        .filter(ProfileTopicPref.profile_id == profile.id)
        .all()
    )

    if not topic_prefs:
        # No preferences set - return None to trigger randomization
        return None

    # Weighted random selection based on preference weights
    topics = [pref.topic for pref in topic_prefs]
    weights = [pref.weight for pref in topic_prefs]

    selected = random.choices(topics, weights=weights, k=1)[0]
    logger.debug(f"Selected topic '{selected}' for profile {profile.id} from preferences")
    return selected


def get_topic_coverage(db: Session, lang: str, target_lang: str) -> Dict[str, int]:
    """Get count of ready texts per topic for a language pair."""
    from server.models import ReadingText

    coverage = {topic: 0 for topic in AVAILABLE_TOPICS}
    coverage["other"] = 0  # For texts with no/unknown topic

    texts = (
        db.query(ReadingText)
        .filter(
            ReadingText.lang == lang,
            ReadingText.target_lang == target_lang,
            ReadingText.words_complete == True,
            ReadingText.sentences_complete == True,
        )
        .all()
    )

    for text in texts:
        if text.topic:
            topics = text.topic.split(",")
            for t in topics:
                t = t.strip()
                if t in coverage:
                    coverage[t] += 1
                else:
                    coverage["other"] += 1
        else:
            coverage["other"] += 1

    return coverage


def select_diverse_topic(db: Session, profile: Profile, preferred_topic: Optional[str] = None) -> str:
    """Select a topic considering both preference and pool diversity.

    Strategy:
    - 70% chance to use preferred topic (if set)
    - 30% chance to pick an underrepresented topic (fewer texts in pool)
    """
    coverage = get_topic_coverage(db, profile.lang, profile.target_lang)

    # Find topics with minimal coverage
    min_coverage = min(coverage.values())
    underrepresented = [t for t, count in coverage.items() if count == min_coverage]

    # Decide whether to use preference or prioritize diversity
    if preferred_topic and random.random() < 0.7:
        # Use preferred topic
        return preferred_topic
    else:
        # Pick from underrepresented topics
        selected = random.choice(underrepresented)
        logger.info(
            f"Diversity selection: chose '{selected}' (has {min_coverage} texts) "
            f"over preference '{preferred_topic}'"
        )
        return selected


def check_urgent_word_coverage(
    db: Session,
    profile: Profile,
    urgent_words: Set[str],
    min_coverage: int = 3,
) -> bool:
    """Check if urgent words are already covered in existing texts.

    Returns True if we have sufficient coverage (don't need to generate more).
    """
    if not urgent_words:
        return False

    from server.models import ReadingWordGloss

    # Count how many urgent words appear in ready texts
    covered_words = set()

    # Check word glosses for ready texts
    glosses = (
        db.query(ReadingWordGloss.lemma)
        .filter(
            ReadingWordGloss.lang == profile.lang,
            ReadingWordGloss.lemma.in_(urgent_words),
        )
        .distinct()
        .all()
    )

    covered_words = {g[0] for g in glosses}

    coverage_ratio = len(covered_words) / len(urgent_words) if urgent_words else 0

    logger.info(
        f"Urgent word coverage: {len(covered_words)}/{len(urgent_words)} "
        f"({coverage_ratio:.1%}) - need {min_coverage} texts per word"
    )

    # If we have good coverage (enough texts contain these words), don't generate
    return coverage_ratio >= 0.5  # At least 50% of urgent words covered


def _ensure_timezone_aware(dt: Optional[datetime]) -> Optional[datetime]:
    """Ensure a datetime is timezone-aware. Convert naive datetimes to UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass
class TextFeatures:
    """Quantified features of a generated text."""

    text_id: int

    # Difficulty and level
    difficulty: float = 0.0  # 0-10 scale

    # Topic distribution (normalized weights)
    topics: Dict[str, float] = field(default_factory=dict)

    # Target words coverage
    target_words_density: Dict[str, float] = field(
        default_factory=dict
    )  # word -> density per 100 words

    # Content metrics
    word_count: int = 0
    unique_lemma_count: int = 0
    avg_word_length: float = 0.0

    # Length ratio (actual / requested in prompt)
    length_ratio: float = 1.0

    # Quality metrics
    rating_avg: float = 0.0
    read_count: int = 0


@dataclass
class TextRequest:
    """User's request for a text with preferences."""

    profile_id: int
    lang: str
    target_lang: str

    # Difficulty preference
    difficulty_target: float = 0.0  # 0-10 scale
    difficulty_tolerance: float = 2.0  # max acceptable diff

    # Topic preferences (topic -> weight)
    topics: Dict[str, float] = field(default_factory=dict)

    # Target words to practice
    target_words: Set[str] = field(default_factory=set)
    target_words_data: List[Dict] = field(default_factory=list)  # Full data with urgency

    # Length preference
    min_length: int = 100
    max_length: int = 500
    preferred_length: int = 250

    # Quality thresholds
    min_rating: float = 0.0  # ignore texts with rating below this


def get_urgent_lexemes_for_profile(
    db: Session,
    profile: Profile,
    limit: int = 20,
    current_time: Optional[datetime] = None,
) -> List[Tuple[Lexeme, float]]:
    """Get lexemes needing review, sorted by urgency.

    Returns list of (lexeme, urgency_score) tuples.
    """
    from server.services.srs import calculate_urgency_score

    if not current_time:
        current_time = datetime.now(timezone.utc)

    lexemes = (
        db.query(Lexeme)
        .filter(
            Lexeme.account_id == profile.account_id,
            Lexeme.profile_id == profile.id,
            Lexeme.lang == profile.lang,
            Lexeme.next_due_at.is_not(None),
        )
        .all()
    )

    scored = [(lex, calculate_urgency_score(lex, current_time)) for lex in lexemes]
    scored.sort(key=lambda x: x[1], reverse=True)

    logger.info(
        f"Found {len(scored)} urgent lexemes for profile {profile.id}, "
        f"returning top {min(limit, len(scored))}"
    )

    return scored[:limit]


def compute_vocabulary_overlap_score(
    db: Session,
    text_id: int,
    urgent_lexemes_with_scores: List[Tuple[Lexeme, float]],
) -> float:
    """Calculate vocabulary overlap score with urgent words.

    Returns 0.0-1.0 where higher = better match.

    Combines coverage (how much urgent vocab is covered) with
    density (urgent words per total unique words).
    """
    glosses = (
        db.query(ReadingWordGloss.lemma, ReadingWordGloss.pos)
        .filter(ReadingWordGloss.text_id == text_id)
        .distinct()
        .all()
    )

    text_vocab = {(lemma, pos) for lemma, pos in glosses}

    if not text_vocab:
        return 0.0

    total_urgency = sum(score for _, score in urgent_lexemes_with_scores)

    if total_urgency == 0:
        return 0.0

    matched_urgency = sum(
        score for (lex, score) in urgent_lexemes_with_scores
        if (lex.lemma, lex.pos) in text_vocab
    )

    coverage = matched_urgency / total_urgency
    density = matched_urgency / max(1, len(text_vocab))

    # Weight coverage 70%, density 30%
    return coverage * 0.7 + density * 0.3


def select_target_words_from_urgent(
    urgent_lexemes: List[Tuple[Lexeme, float]],
    count: int = 5,
) -> List[Dict]:
    """Select diverse target words from urgent lexemes.

    Strategy:
    - Pick top N words by urgency
    - Diversify by POS (avoid 5 nouns)
    - Prefer words with higher variance (more uncertain)

    Returns list of dicts with {lemma, pos, urgency}
    """
    selected = []
    pos_counts = {}

    for lexeme, urgency in urgent_lexemes:
        if len(selected) >= count:
            break

        pos = lexeme.pos or "UNKNOWN"

        # Limit POS diversity (max 2 of same POS)
        if pos_counts.get(pos, 0) >= 2:
            continue

        selected.append({
            "lemma": lexeme.lemma,
            "pos": pos,
            "urgency": urgency,
        })
        pos_counts[pos] = pos_counts.get(pos, 0) + 1

    logger.info(
        f"Selected {len(selected)} target words from {len(urgent_lexemes)} urgent lexemes: "
        f"{[w['lemma'] for w in selected]}"
    )

    return selected


def compute_text_features(db: Session, text: ReadingText) -> TextFeatures:
    """Extract features from a ReadingText."""
    try:
        features = TextFeatures(text_id=text.id)

        # Difficulty from ci_target (normalize 0.85-0.95 to 0-10)
        if text.ci_target:
            features.difficulty = (0.95 - text.ci_target) / 0.01  # 0.95->0, 0.85->10
        elif text.difficulty_estimate:
            features.difficulty = text.difficulty_estimate

        # Topic from text.topic
        if text.topic:
            topic_map = {
                "fiction": "fiction",
                "news": "news",
                "science": "science",
                "technology": "technology",
                "history": "history",
                "daily_life": "daily_life",
                "culture": "culture",
                "sports": "sports",
                "business": "business",
            }
            topics_str = text.topic
            for topic in topics_str.split(","):
                topic = topic.strip()
                if topic in topic_map:
                    features.topics[topic_map[topic]] = 1.0

        # Word count metrics
        features.word_count = text.word_count or 0
        features.unique_lemma_count = text.unique_lemma_count or 0

        # Target words from prompt_words
        if text.prompt_words and isinstance(text.prompt_words, dict):
            prompt_words = text.prompt_words.get("words", [])
            if features.word_count > 0:
                for word in prompt_words:
                    density = (prompt_words.count(word) / features.word_count) * 100
                    features.target_words_density[word] = density

        # Length ratio from prompt_level_hint or default
        if features.word_count > 0:
            preferred = 250  # default
            if text.prompt_level_hint:
                try:
                    parts = text.prompt_level_hint.split()
                    if "words" in text.prompt_level_hint:
                        preferred = int(parts[-2])
                except (ValueError, IndexError):
                    pass
            features.length_ratio = (
                features.word_count / preferred if preferred > 0 else 1.0
            )

        # Quality metrics
        features.rating_avg = text.rating_avg or 0.0
        if text.rating_avg:
            read_count = (
                db.query(ProfileTextRead)
                .filter(ProfileTextRead.text_id == text.id)
                .count()
            )
            features.read_count = read_count

        return features

    except Exception as e:
        logger.error(f"Error computing features for text {text.id}: {e}")
        return TextFeatures(text_id=text.id)


def build_profile_request(db: Session, profile: Profile) -> TextRequest:
    """Build a text request from a user profile."""
    try:
        request = TextRequest(
            profile_id=profile.id,
            lang=profile.lang,
            target_lang=profile.target_lang,
            difficulty_target=profile.level_value,
            difficulty_tolerance=max(1.0, profile.level_var),  # at least 1.0
        )

        # Topic preferences
        topic_prefs = (
            db.query(ProfileTopicPref)
            .filter(ProfileTopicPref.profile_id == profile.id)
            .all()
        )
        if topic_prefs:
            total_weight = sum(p.weight for p in topic_prefs)
            if total_weight > 0:
                for pref in topic_prefs:
                    request.topics[pref.topic] = pref.weight / total_weight

        # Length preference
        if profile.text_length:
            request.preferred_length = profile.text_length
            request.min_length = int(profile.text_length * 0.5)
            request.max_length = int(profile.text_length * 1.5)

        # Target words from SRS (urgent lexemes)
        # Reduced count and randomized to add more variety
        import random
        urgent_lexemes = get_urgent_lexemes_for_profile(db, profile, limit=20)
        if urgent_lexemes and random.random() < 0.7:  # 70% chance to include target words
            # Use fewer target words (2-3) for more variety
            target_count = random.choice([0, 2, 3])  # Sometimes 0 (no specific targets)
            if target_count > 0:
                target_words_data = select_target_words_from_urgent(urgent_lexemes, count=target_count)
                request.target_words = {w["lemma"] for w in target_words_data}
                request.target_words_data = target_words_data
                logger.info(
                    f"Profile {profile.id}: target_words={request.target_words}, "
                    f"count={len(request.target_words)}"
                )
            else:
                request.target_words = set()
                request.target_words_data = []
                logger.debug(f"Profile {profile.id}: no target words (randomized)")
        else:
            request.target_words = set()
            request.target_words_data = []

        return request

    except Exception as e:
        logger.error(f"Error building request for profile {profile.id}: {e}")
        return TextRequest(
            profile_id=profile.id,
            lang=profile.lang,
            target_lang=profile.target_lang,
        )


def compute_similarity_score(
    text_features: TextFeatures,
    request: TextRequest,
    urgency_vocab_score: float = 0.0,
) -> float:
    """Compute similarity score between text and request.

    Returns a distance score (lower = better match).
    """
    try:
        total_score = 0.0

        # 1. Difficulty distance (weight: 3.0) - PRIMARY FACTOR
        diff_distance = abs(text_features.difficulty - request.difficulty_target)
        if diff_distance > request.difficulty_tolerance:
            diff_penalty = 100.0  # Outside tolerance, very bad
        else:
            diff_penalty = diff_distance * 3.0
        total_score += diff_penalty

        # 2. Topic similarity (weight: 2.0)
        if request.topics:
            if text_features.topics:
                # Jaccard similarity
                request_topics = set(request.topics.keys())
                text_topics = set(text_features.topics.keys())
                intersection = request_topics & text_topics
                union = request_topics | text_topics
                jaccard = len(intersection) / len(union) if union else 0.0
                topic_penalty = (1.0 - jaccard) * 2.0
            else:
                topic_penalty = 2.0  # No topics in text
            total_score += topic_penalty

        # 3. Length fit (weight: 1.0)
        if text_features.word_count < request.min_length:
            length_penalty = (request.min_length - text_features.word_count) / 10.0
        elif text_features.word_count > request.max_length:
            length_penalty = (text_features.word_count - request.max_length) / 10.0
        else:
            # Optimal length has zero penalty
            length_penalty = (
                abs(text_features.word_count - request.preferred_length) / 100.0
            )
        total_score += length_penalty

        # 4. Target word coverage (weight: 0.5) - REDUCED from 1.5
        # This is a "nice to have", not a requirement
        if request.target_words:
            covered_words = (
                request.target_words & text_features.target_words_density.keys()
            )
            coverage = (
                len(covered_words) / len(request.target_words)
                if request.target_words
                else 0.0
            )
            word_penalty = (1.0 - coverage) * 0.5
            total_score += word_penalty

        # 5. Urgency vocabulary overlap (weight: 0.5) - REDUCED from 2.5
        # This is a bonus factor, should not drive generation
        if urgency_vocab_score > 0:
            # Invert: high overlap = low penalty (bonus)
            urgency_penalty = (1.0 - urgency_vocab_score) * 0.5
            total_score += urgency_penalty

        # 6. Quality bonus (negative penalty = reward)
        # Binary scale: rating_avg from -1.0 to 1.0
        if text_features.rating_avg >= 0.5:  # 75%+ likes
            total_score -= 0.5
        elif text_features.rating_avg >= 0.0:  # 50%+ likes
            total_score -= 0.2
        elif text_features.rating_avg <= -0.5:  # 75%+ dislikes - penalize
            total_score += 1.0

        return total_score

    except Exception as e:
        logger.error(f"Error computing similarity score: {e}")
        return 100.0  # Return worst score on error


def get_unread_texts_for_profile(
    db: Session, profile: Profile, limit: int = 50
) -> List[Tuple[ReadingText, float]]:
    """Get scored unread texts for a profile."""
    try:
        # Get read text IDs
        read_ids = {
            ptr.text_id
            for ptr in db.query(ProfileTextRead)
            .filter(ProfileTextRead.profile_id == profile.id)
            .all()
        }

        # Check reread cooldown
        if profile.reread_cooldown_days is not None:
            # None = never, 0 = always allow
            # Use naive UTC datetime for SQLite compatibility
            cutoff_date_naive = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
                days=profile.reread_cooldown_days
            )
            recent_reads = (
                db.query(ProfileTextRead)
                .filter(
                    ProfileTextRead.profile_id == profile.id,
                    ProfileTextRead.last_read_at >= cutoff_date_naive,
                )
                .all()
            )
            read_ids.update(ptr.text_id for ptr in recent_reads)

        # Get ready texts that haven't been read
        available_texts = (
            db.query(ReadingText)
            .filter(
                ReadingText.lang == profile.lang,
                ReadingText.target_lang == profile.target_lang,
                ReadingText.content.is_not(None),
                ReadingText.content != "",
                ReadingText.words_complete == True,
                ReadingText.sentences_complete == True,
                ReadingText.is_hidden == False,
                ~ReadingText.id.in_(read_ids),
            )
            .limit(limit)
            .all()
        )

        # Build request
        request = build_profile_request(db, profile)

        # Get urgent lexemes for vocabulary matching
        urgent_lexemes = get_urgent_lexemes_for_profile(db, profile, limit=10)

        # Score all texts
        scored_texts = []
        for text in available_texts:
            features = compute_text_features(db, text)

            # Calculate vocabulary overlap with urgent words
            vocab_overlap = 0.0
            if urgent_lexemes:
                vocab_overlap = compute_vocabulary_overlap_score(
                    db, text.id, urgent_lexemes
                )

            score = compute_similarity_score(
                features, request, urgency_vocab_score=vocab_overlap
            )
            scored_texts.append((text, score))

        # Sort by score (lowest = best)
        scored_texts.sort(key=lambda x: x[1])

        return scored_texts

    except Exception as e:
        logger.error(f"Error getting unread texts for profile {profile.id}: {e}")
        return []


def select_best_text(db: Session, profile: Profile) -> Optional[ReadingText]:
    """Select the best text for a profile from the pool."""
    try:
        scored_texts = get_unread_texts_for_profile(db, profile, limit=50)

        if not scored_texts:
            logger.info(f"No available texts for profile {profile.id}")
            return None

        best_text, best_score = scored_texts[0]
        logger.info(
            f"Selected text {best_text.id} for profile {profile.id} with score {best_score:.2f}"
        )

        return best_text

    except Exception as e:
        logger.error(f"Error selecting best text for profile {profile.id}: {e}")
        return None


def update_text_queue(db: Session, profile: Profile) -> None:
    """Update the cached text queue for a profile."""
    try:
        scored_texts = get_unread_texts_for_profile(db, profile, limit=50)

        # Clear old queue
        db.query(ProfileTextQueue).filter(
            ProfileTextQueue.profile_id == profile.id
        ).delete()

        # Insert new queue entries
        for rank, (text, score) in enumerate(scored_texts[:20]):  # Keep top 20
            queue_entry = ProfileTextQueue(
                profile_id=profile.id,
                text_id=text.id,
                rank=rank,
                score=score,
            )
            db.add(queue_entry)

        db.commit()
        logger.info(
            f"Updated text queue for profile {profile.id} with {len(scored_texts)} texts"
        )

    except Exception as e:
        logger.error(f"Error updating text queue for profile {profile.id}: {e}")
        db.rollback()


def get_text_from_queue(db: Session, profile: Profile) -> Optional[ReadingText]:
    """Get the next text from the cached queue."""
    try:
        # Get highest-ranked unread text from queue
        queue_entry = (
            db.query(ProfileTextQueue)
            .filter(ProfileTextQueue.profile_id == profile.id)
            .order_by(ProfileTextQueue.rank)
            .first()
        )

        if not queue_entry:
            # Queue empty, refresh it
            update_text_queue(db, profile)
            queue_entry = (
                db.query(ProfileTextQueue)
                .filter(ProfileTextQueue.profile_id == profile.id)
                .order_by(ProfileTextQueue.rank)
                .first()
            )

        if not queue_entry:
            return None

        text = (
            db.query(ReadingText).filter(ReadingText.id == queue_entry.text_id).first()
        )
        return text

    except Exception as e:
        logger.error(f"Error getting text from queue for profile {profile.id}: {e}")
        return None


def detect_pool_gaps(db: Session, threshold: float = 3.0) -> List[TextRequest]:
    """Detect gaps in the text pool where generation is needed.

    Only generates if:
    1. No texts exist for this language pair, OR
    2. Best matching text score > threshold AND we have minimal coverage
    3. Urgent words are NOT already covered in existing texts (fixes urgency loop)

    Returns list of TextRequest objects representing missing content.
    """
    try:
        gaps = []

        # Get all active profiles
        profiles = db.query(Profile).filter(Profile.preferences_updating == False).all()

        for profile in profiles:
            request = build_profile_request(db, profile)
            scored_texts = get_unread_texts_for_profile(db, profile, limit=50)

            if not scored_texts:
                # No texts at all for this language pair
                logger.info(f"[Gap] No texts for {profile.lang}->{profile.target_lang}")
                gaps.append(request)
                continue

            # Check if best match is good enough
            best_score = scored_texts[0][1]

            # Also check if we have minimal coverage (at least 5 decent texts with score < 5.0)
            decent_match_count = sum(1 for _, score in scored_texts if score < 5.0)

            # Check if urgent words are already covered (fixes urgency loop)
            urgent_words_covered = False
            if request.target_words:
                urgent_words_covered = check_urgent_word_coverage(
                    db, profile, request.target_words
                )

            if best_score > threshold and decent_match_count < 5:
                # Pool has texts but insufficient good coverage
                if urgent_words_covered:
                    logger.info(
                        f"[NoGap] Urgent words already covered, skipping generation "
                        f"despite score {best_score:.2f}"
                    )
                else:
                    logger.info(
                        f"[Gap] Best score {best_score:.2f} > {threshold}, "
                        f"only {decent_match_count} decent texts for {profile.lang}"
                    )
                    gaps.append(request)
            else:
                logger.debug(
                    f"[NoGap] Best score {best_score:.2f}, {decent_match_count} decent texts"
                )

        return gaps

    except Exception as e:
        logger.error(f"Error detecting pool gaps: {e}")
        return []
