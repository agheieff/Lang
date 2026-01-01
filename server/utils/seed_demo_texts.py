"""
Seed demo texts with translations for each supported language.
Run this with: python -m server.utils.seed_demo_texts
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy.orm import Session
from server.db import get_db
from server.models import ReadingText, ReadingTextTranslation, ReadingWordGloss


def seed_spanish_demo(db: Session) -> None:
    """Seed Spanish demo text with translations."""

    # Check if demo already exists
    existing = (
        db.query(ReadingText)
        .filter(
            ReadingText.lang == "es",
            ReadingText.target_lang == "en",
            ReadingText.source == "demo",
        )
        .first()
    )
    if existing:
        print("  Spanish demo already exists, skipping...")
        return

    # Create the reading text
    text = ReadingText(
        lang="es",
        target_lang="en",
        content="¡Hola! Bienvenido a tu práctica de lectura. Este es un texto de demostración simple para comenzar. El sistema completo de generación de textos estará disponible pronto.",
        title=None,
        source="demo",
        words_complete=True,
        sentences_complete=True,
        ci_target=0.8,
        topic="daily_life",
        difficulty_estimate=0.3,
        word_count=27,
        unique_lemma_count=24,
    )
    db.add(text)
    db.flush()

    # Add sentence translations
    sentences = [
        "¡Hola! Bienvenido a tu práctica de lectura.",
        "Este es un texto de demostración simple para comenzar.",
        "El sistema completo de generación de textos estará disponible pronto.",
    ]

    for idx, source in enumerate(sentences):
        translation = ReadingTextTranslation(
            text_id=text.id,
            target_lang="en",
            unit="sentence",
            segment_index=idx,
            source_text=source,
            translated_text="",
            provider="manual",
        )
        db.add(translation)

    # Add word translations (span_start, span_end based on content string)
    words_data = [
        # Word, Lemma, POS, Translation, Start, End
        ("¡Hola!", "hola", "INTJ", "Hello!", 0, 6),
        ("Bienvenido", "bienvenido", "ADJ", "Welcome", 7, 17),
        ("a", "a", "ADP", "to", 18, 19),
        ("tu", "tú", "PRON", "your", 20, 22),
        ("práctica", "práctica", "NOUN", "practice", 23, 31),
        ("de", "de", "ADP", "of", 32, 34),
        ("lectura", "lectura", "NOUN", "reading", 35, 42),
        ("Este", "este", "DET", "This", 44, 48),
        ("es", "ser", "AUX", "is", 49, 51),
        ("un", "uno", "DET", "a", 52, 54),
        ("texto", "texto", "NOUN", "text", 55, 60),
        ("de", "de", "ADP", "of", 61, 63),
        ("demostración", "demostración", "NOUN", "demonstration", 64, 76),
        ("simple", "simple", "ADJ", "simple", 77, 83),
        ("para", "para", "ADP", "to", 84, 88),
        ("comenzar", "comenzar", "VERB", "get started", 89, 97),
        ("El", "el", "DET", "The", 99, 101),
        ("sistema", "sistema", "NOUN", "system", 102, 109),
        ("completo", "completo", "ADJ", "complete", 110, 118),
        ("de", "de", "ADP", "of", 119, 121),
        ("generación", "generación", "NOUN", "generation", 122, 132),
        ("de", "de", "ADP", "of", 133, 135),
        ("textos", "texto", "NOUN", "texts", 136, 142),
        ("estará", "estar", "AUX", "will be", 143, 149),
        ("disponible", "disponible", "ADJ", "available", 150, 160),
        ("pronto", "pronto", "ADV", "soon", 161, 167),
    ]

    for surface, lemma, pos, translation, start, end in words_data:
        word = ReadingWordGloss(
            text_id=text.id,
            target_lang="en",
            lang="es",
            surface=surface,
            lemma=lemma,
            pos=pos,
            translation=translation,
            lemma_translation=translation,
            span_start=start,
            span_end=end,
        )
        db.add(word)

    print("  ✓ Spanish demo text created")


def seed_chinese_simplified_demo(db: Session) -> None:
    """Seed Chinese Simplified demo text with translations."""

    existing = (
        db.query(ReadingText)
        .filter(
            ReadingText.lang == "zh-CN",
            ReadingText.target_lang == "en",
            ReadingText.source == "demo",
        )
        .first()
    )
    if existing:
        print("  Chinese Simplified demo already exists, skipping...")
        return

    text = ReadingText(
        lang="zh-CN",
        target_lang="en",
        content="你好！欢迎来到阅读练习。这是一个简单的演示文本，帮你开始学习。完整的文本生成系统很快就会推出。",
        title=None,
        source="demo",
        words_complete=True,
        sentences_complete=True,
        ci_target=0.8,
        topic="daily_life",
        difficulty_estimate=0.3,
        word_count=28,
        unique_lemma_count=25,
    )
    db.add(text)
    db.flush()

    sentences = [
        "你好！欢迎来到阅读练习。",
        "这是一个简单的演示文本，帮你开始学习。",
        "完整的文本生成系统很快就会推出。",
    ]

    for idx, source in enumerate(sentences):
        translation = ReadingTextTranslation(
            text_id=text.id,
            target_lang="en",
            unit="sentence",
            segment_index=idx,
            source_text=source,
            translated_text="",
            provider="manual",
        )
        db.add(translation)

    # Calculate character positions for Chinese
    content = text.content
    words_data = [
        ("你好", "你好", "INTJ", "Hello", 0, 2, "nǐ hǎo"),
        ("欢迎", "欢迎", "VERB", "welcome", 3, 5, "huān yíng"),
        ("来到", "来到", "VERB", "come to", 6, 8, "lái dào"),
        ("阅读", "阅读", "NOUN", "reading", 9, 11, "yuè dú"),
        ("练习", "练习", "NOUN", "practice", 12, 14, "liàn xí"),
        ("这是", "这是", "PRON", "this is", 15, 17, "zhè shì"),
        ("一个", "一个", "DET", "a", 18, 20, "yí gè"),
        ("简单", "简单", "ADJ", "simple", 21, 23, "jiǎn dān"),
        ("的", "的", "PART", "of", 24, 25, "de"),
        ("演示", "演示", "NOUN", "demonstration", 26, 28, "yǎn shì"),
        ("文本", "文本", "NOUN", "text", 29, 31, "wén běn"),
        ("帮", "帮", "VERB", "help", 33, 34, "bāng"),
        ("你", "你", "PRON", "you", 35, 36, "nǐ"),
        ("开始", "开始", "VERB", "start", 37, 39, "kāi shǐ"),
        ("学习", "学习", "VERB", "learning", 40, 42, "xué xí"),
        ("完整", "完整", "ADJ", "complete", 44, 46, "wán zhěng"),
        ("的", "的", "PART", "of", 47, 48, "de"),
        ("文本", "文本", "NOUN", "text", 49, 51, "wén běn"),
        ("生成", "生成", "NOUN", "generation", 52, 54, "shēng chéng"),
        ("系统", "系统", "NOUN", "system", 55, 57, "xì tǒng"),
        ("很快", "很快", "ADV", "soon", 58, 60, "hěn kuài"),
        ("就", "就", "ADV", "will", 61, 62, "jiù"),
        ("会", "会", "AUX", "will", 63, 64, "huì"),
        ("推出", "推出", "VERB", "launch", 65, 67, "tuī chū"),
    ]

    for surface, lemma, pos, translation, start, end, pinyin in words_data:
        word = ReadingWordGloss(
            text_id=text.id,
            target_lang="en",
            lang="zh-CN",
            surface=surface,
            lemma=lemma,
            pos=pos,
            pinyin=pinyin,
            translation=translation,
            lemma_translation=translation,
            span_start=start,
            span_end=end,
        )
        db.add(word)

    print("  ✓ Chinese Simplified demo text created")


def seed_chinese_traditional_demo(db: Session) -> None:
    """Seed Chinese Traditional demo text with translations."""

    existing = (
        db.query(ReadingText)
        .filter(
            ReadingText.lang == "zh-TW",
            ReadingText.target_lang == "en",
            ReadingText.source == "demo",
        )
        .first()
    )
    if existing:
        print("  Chinese Traditional demo already exists, skipping...")
        return

    text = ReadingText(
        lang="zh-TW",
        target_lang="en",
        content="你好！歡迎來到閱讀練習。這是一個簡單的演示文本，幫你開始學習。完整的文本生成系統很快就會推出。",
        title=None,
        source="demo",
        words_complete=True,
        sentences_complete=True,
        ci_target=0.8,
        topic="daily_life",
        difficulty_estimate=0.3,
        word_count=28,
        unique_lemma_count=25,
    )
    db.add(text)
    db.flush()

    sentences = [
        "你好！歡迎來到閱讀練習。",
        "這是一個簡單的演示文本，幫你開始學習。",
        "完整的文本生成系統很快就會推出。",
    ]

    for idx, source in enumerate(sentences):
        translation = ReadingTextTranslation(
            text_id=text.id,
            target_lang="en",
            unit="sentence",
            segment_index=idx,
            source_text=source,
            translated_text="",
            provider="manual",
        )
        db.add(translation)

    content = text.content
    words_data = [
        ("你好", "你好", "INTJ", "Hello", 0, 2, "nǐ hǎo"),
        ("歡迎", "歡迎", "VERB", "welcome", 3, 5, "huān yíng"),
        ("來到", "來到", "VERB", "come to", 6, 8, "lái dào"),
        ("閱讀", "閱讀", "NOUN", "reading", 9, 11, "yuè dú"),
        ("練習", "練習", "NOUN", "practice", 12, 14, "liàn xí"),
        ("這是", "這是", "PRON", "this is", 15, 17, "zhè shì"),
        ("一個", "一個", "DET", "a", 18, 20, "yí gè"),
        ("簡單", "簡單", "ADJ", "simple", 21, 23, "jiǎn dān"),
        ("的", "的", "PART", "of", 24, 25, "de"),
        ("演示", "演示", "NOUN", "demonstration", 26, 28, "yǎn shì"),
        ("文本", "文本", "NOUN", "text", 29, 31, "wén běn"),
        ("幫", "幫", "VERB", "help", 33, 34, "bāng"),
        ("你", "你", "PRON", "you", 35, 36, "nǐ"),
        ("開始", "開始", "VERB", "start", 37, 39, "kāi shǐ"),
        ("學習", "學習", "VERB", "learning", 40, 42, "xué xí"),
        ("完整", "完整", "ADJ", "complete", 44, 46, "wán zhěng"),
        ("的", "的", "PART", "of", 47, 48, "de"),
        ("文本", "文本", "NOUN", "text", 49, 51, "wén běn"),
        ("生成", "生成", "NOUN", "generation", 52, 54, "shēng chéng"),
        ("系統", "系統", "NOUN", "system", 55, 57, "xì tǒng"),
        ("很快", "很快", "ADV", "soon", 58, 60, "hěn kuài"),
        ("就", "就", "ADV", "will", 61, 62, "jiù"),
        ("會", "會", "AUX", "will", 63, 64, "huì"),
        ("推出", "推出", "VERB", "launch", 65, 67, "tuī chū"),
    ]

    for surface, lemma, pos, translation, start, end, pinyin in words_data:
        word = ReadingWordGloss(
            text_id=text.id,
            target_lang="en",
            lang="zh-TW",
            surface=surface,
            lemma=lemma,
            pos=pos,
            pinyin=pinyin,
            translation=translation,
            lemma_translation=translation,
            span_start=start,
            span_end=end,
        )
        db.add(word)

    print("  ✓ Chinese Traditional demo text created")


def seed_english_demo(db: Session) -> None:
    """Seed English demo text for Spanish speakers."""

    existing = (
        db.query(ReadingText)
        .filter(
            ReadingText.lang == "en",
            ReadingText.target_lang == "es",
            ReadingText.source == "demo",
        )
        .first()
    )
    if existing:
        print("  English demo already exists, skipping...")
        return

    text = ReadingText(
        lang="en",
        target_lang="es",
        content="Hello! Welcome to your reading practice. This is a simple demo text to get you started. The full text generation system will be available soon.",
        title=None,
        source="demo",
        words_complete=True,
        sentences_complete=True,
        ci_target=0.8,
        topic="daily_life",
        difficulty_estimate=0.3,
        word_count=27,
        unique_lemma_count=24,
    )
    db.add(text)
    db.flush()

    sentences = [
        "Hello! Welcome to your reading practice.",
        "This is a simple demo text to get you started.",
        "The full text generation system will be available soon.",
    ]

    for idx, source in enumerate(sentences):
        translation = ReadingTextTranslation(
            text_id=text.id,
            target_lang="es",
            unit="sentence",
            segment_index=idx,
            source_text=source,
            translated_text="",
            provider="manual",
        )
        db.add(translation)

    content = text.content
    words_data = [
        ("Hello!", "hello", "INTJ", "¡Hola!", 0, 6),
        ("Welcome", "welcome", "INTJ", "Bienvenido", 7, 15),
        ("to", "to", "ADP", "a", 16, 18),
        ("your", "your", "DET", "tu", 19, 23),
        ("reading", "reading", "NOUN", "lectura", 24, 31),
        ("practice", "practice", "NOUN", "práctica", 32, 40),
        ("This", "this", "DET", "Este", 42, 46),
        ("is", "be", "AUX", "es", 47, 49),
        ("a", "a", "DET", "un", 50, 51),
        ("simple", "simple", "ADJ", "simple", 52, 58),
        ("demo", "demo", "NOUN", "demostración", 59, 63),
        ("text", "text", "NOUN", "texto", 64, 68),
        ("to", "to", "ADP", "para", 69, 71),
        ("get", "get", "VERB", "comenzar", 72, 75),
        ("you", "you", "PRON", "te", 76, 79),
        ("started", "start", "VERB", "empezado", 80, 88),
        ("The", "the", "DET", "El", 90, 93),
        ("full", "full", "ADJ", "completo", 94, 98),
        ("text", "text", "NOUN", "texto", 99, 103),
        ("generation", "generation", "NOUN", "generación", 104, 114),
        ("system", "system", "NOUN", "sistema", 115, 121),
        ("will", "will", "AUX", "estará", 122, 126),
        ("be", "be", "AUX", "será", 127, 129),
        ("available", "available", "ADJ", "disponible", 130, 139),
        ("soon", "soon", "ADV", "pronto", 140, 144),
    ]

    for surface, lemma, pos, translation, start, end in words_data:
        word = ReadingWordGloss(
            text_id=text.id,
            target_lang="es",
            lang="en",
            surface=surface,
            lemma=lemma,
            pos=pos,
            translation=translation,
            lemma_translation=translation,
            span_start=start,
            span_end=end,
        )
        db.add(word)

    print("  ✓ English demo text created")


def seed_all_demo_texts() -> None:
    """Seed all demo texts for supported languages."""

    db = next(get_db())

    try:
        print("Seeding demo texts...")

        print("\n[Spanish (es) → English]")
        seed_spanish_demo(db)

        print("\n[Chinese Simplified (zh-CN) → English]")
        seed_chinese_simplified_demo(db)

        print("\n[Chinese Traditional (zh-TW) → English]")
        seed_chinese_traditional_demo(db)

        print("\n[English (en) → Spanish]")
        seed_english_demo(db)

        db.commit()
        print("\n✓ All demo texts seeded successfully!")

    except Exception as e:
        db.rollback()
        print(f"\n✗ Error seeding demo texts: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_all_demo_texts()
