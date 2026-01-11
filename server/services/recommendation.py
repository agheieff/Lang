"""Text recommendation engine for matching texts to user profiles."""

from __future__ import annotations

import logging
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
        urgent_lexemes = get_urgent_lexemes_for_profile(db, profile, limit=20)
        if urgent_lexemes:
            request.target_words = {lex.lemma for lex, _ in urgent_lexemes[:5]}
            request.target_words_data = [
                {"lemma": lex.lemma, "pos": lex.pos, "urgency": urgency}
                for lex, urgency in urgent_lexemes[:5]
            ]
            logger.info(
                f"Profile {profile.id}: target_words={request.target_words}, "
                f"count={len(request.target_words)}"
            )
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

        # 1. Difficulty distance (weight: 3.0)
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

        # 4. Target word coverage (weight: 1.5)
        if request.target_words:
            covered_words = (
                request.target_words & text_features.target_words_density.keys()
            )
            coverage = (
                len(covered_words) / len(request.target_words)
                if request.target_words
                else 0.0
            )
            word_penalty = (1.0 - coverage) * 1.5
            total_score += word_penalty

        # 5. Urgency vocabulary overlap (weight: 2.5)
        if urgency_vocab_score > 0:
            # Invert: high overlap = low penalty
            urgency_penalty = (1.0 - urgency_vocab_score) * 2.5
            total_score += urgency_penalty

        # 6. Quality bonus (negative penalty = reward)
        if text_features.rating_avg >= 4.0:
            total_score -= 0.5
        elif text_features.rating_avg >= 3.0:
            total_score -= 0.2

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
            cutoff_date = datetime.now(timezone.utc) - timedelta(
                days=profile.reread_cooldown_days
            )
            recent_reads = (
                db.query(ProfileTextRead)
                .filter(
                    ProfileTextRead.profile_id == profile.id,
                    ProfileTextRead.last_read_at >= cutoff_date,
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
                gaps.append(request)
                continue

            # Check if best match is good enough
            best_score = scored_texts[0][1]
            if best_score > threshold:
                # Pool has texts but none match well enough
                gaps.append(request)

        return gaps

    except Exception as e:
        logger.error(f"Error detecting pool gaps: {e}")
        return []
