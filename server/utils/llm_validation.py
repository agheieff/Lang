"""
Pydantic models for LLM response validation.

Provides structured validation for LLM responses to prevent runtime errors.
"""

import logging
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator, constr

logger = logging.getLogger(__name__)


class WordTranslation(BaseModel):
    """Model for individual word translations."""

    translation: str
    lemma: Optional[str] = None
    lemma_translation: Optional[str] = None
    pos: Optional[str] = None
    pinyin: Optional[str] = None
    grammar: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("translation")
    @classmethod
    def validate_translation(cls, v):
        if not v or not v.strip():
            raise ValueError("Translation cannot be empty")
        return v.strip()


class SentenceTranslation(BaseModel):
    """Model for sentence translations."""

    translation: str
    source_text: str

    @field_validator("translation")
    @classmethod
    def validate_translation(cls, v):
        if not v or not v.strip():
            raise ValueError("Translation cannot be empty")
        return v.strip()

    @field_validator("source_text")
    @classmethod
    def validate_source_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Source text cannot be empty")
        return v.strip()


class ParagraphTranslation(BaseModel):
    """Model for paragraph translations."""

    translation: str
    source_text: str

    @field_validator("translation")
    @classmethod
    def validate_translation(cls, v):
        if not v or not v.strip():
            raise ValueError("Translation cannot be empty")
        return v.strip()


class TextTranslation(BaseModel):
    """Model for full text translations."""

    translation: str
    source_text: str

    @field_validator("translation")
    @classmethod
    def validate_translation(cls, v):
        if not v or not v.strip():
            raise ValueError("Translation cannot be empty")
        return v.strip()


class WordTranslationResponse(BaseModel):
    """Response model for word translation requests."""

    words: List[WordTranslation]
    success: bool = True
    error: Optional[str] = None

    @field_validator("words")
    @classmethod
    def validate_words(cls, v):
        if not v:
            raise ValueError("At least one word translation is required")
        return v


class SentenceTranslationResponse(BaseModel):
    """Response model for sentence translation requests."""

    sentences: List[SentenceTranslation]
    success: bool = True
    error: Optional[str] = None


class StructuredTranslationResponse(BaseModel):
    """Response model for structured translation (mixed units)."""

    items: List[Union[SentenceTranslation, ParagraphTranslation, TextTranslation]]
    success: bool = True
    error: Optional[str] = None


class TextGenerationResponse(BaseModel):
    """Response model for text generation requests."""

    content: str
    title: Optional[str] = None
    word_count: Optional[int] = Field(None, ge=0)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v.strip()


class LLMError(BaseModel):
    """Model for LLM error responses."""

    error: str
    error_type: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    success: bool = False


class ParsedLLMResponse(BaseModel):
    """Generic parsed LLM response with validation metadata."""

    raw_content: str
    parsed_content: Union[
        WordTranslationResponse,
        SentenceTranslationResponse,
        StructuredTranslationResponse,
        TextGenerationResponse,
        LLMError,
    ]
    validation_errors: List[str] = Field(default_factory=list)
    parser_version: str = "1.0"


class TranslationRequest(BaseModel):
    """Request model for translation operations."""

    text: str
    source_lang: str
    target_lang: str
    unit: constr(to_upper=True, pattern="^(word|sentence|paragraph|text)$")
    context: Optional[str] = None
    word_context: Dict[str, Any] | None = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class GenerationRequest(BaseModel):
    """Request model for text generation operations."""

    source_lang: str
    target_lang: str
    level: Optional[float] = Field(None, ge=0.0, le=10.0)
    length: Optional[int] = Field(None, gt=0, le=5000)
    topic: Optional[str] = None
    prompt_overrides: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_lang")
    @classmethod
    def validate_languages(cls, v, values):
        if "source_lang" in values and v == values["source_lang"]:
            raise ValueError("Source and target languages must be different")
        return v


# Validation utilities
class ResponseValidator:
    """Utility class for validating LLM responses."""

    @staticmethod
    def validate_word_translation(text: str) -> WordTranslationResponse:
        """Validate and parse word translation response."""
        try:
            # Try to parse as JSON first
            import json

            data = json.loads(text)
            return WordTranslationResponse(**data)
        except Exception as e:
            logger.warning(f"Failed to parse word translation as JSON: {e}")

            # Fallback to structured extraction
            from server.utils.nlp import extract_word_translations

            words = extract_word_translations(text)

            return WordTranslationResponse(
                words=[
                    WordTranslation(
                        translation=w.get("translation", ""),
                        lemma=w.get("lemma"),
                        lemma_translation=w.get("lemma_translation"),
                        pos=w.get("pos"),
                        pinyin=w.get("pinyin"),
                        grammar=w.get("grammar", {}),
                    )
                    for w in words
                ]
            )

    @staticmethod
    def validate_sentence_translation(text: str) -> SentenceTranslationResponse:
        """Validate and parse sentence translation response."""
        try:
            # Try to parse as JSON first
            import json

            data = json.loads(text)
            return SentenceTranslationResponse(**data)
        except Exception as e:
            logger.warning(f"Failed to parse sentence translation as JSON: {e}")

            # Fallback to structured extraction
            from server.utils.nlp import extract_structured_translation

            translation_data = extract_structured_translation(text)
            translation_str = (
                str(translation_data.get("translation", ""))
                if isinstance(translation_data, dict)
                else str(translation_data)
            )

            return SentenceTranslationResponse(
                sentences=[
                    SentenceTranslation(translation=translation_str, source_text="")
                ]
            )

    @staticmethod
    def validate_text_generation(text: str) -> TextGenerationResponse:
        """Validate and parse text generation response."""
        try:
            # Try to parse as JSON first
            import json

            data = json.loads(text)
            return TextGenerationResponse(**data)
        except Exception as e:
            logger.warning(f"Failed to parse text generation as JSON: {e}")

            # Fallback to plain text
            from server.utils.nlp import extract_text_from_llm_response

            content = extract_text_from_llm_response(text)

            return TextGenerationResponse(
                content=content, word_count=len(content.split())
            )

    @staticmethod
    def validate_error_response(text: str) -> LLMError:
        """Validate and parse error response."""
        try:
            # Try to parse as JSON first
            import json

            data = json.loads(text)
            return LLMError(**data)
        except Exception:
            # Fallback to plain text error
            return LLMError(error=text.strip(), error_type="unknown")


# Response extraction utilities
class ResponseExtractor:
    """Utility class for extracting structured data from LLM responses."""

    @staticmethod
    def extract_translations(
        text: str, unit: str = "word"
    ) -> Union[WordTranslationResponse, SentenceTranslationResponse, LLMError]:
        """Extract translations based on unit type."""
        if unit == "word":
            return ResponseValidator.validate_word_translation(text)
        elif unit == "sentence":
            return ResponseValidator.validate_sentence_translation(text)
        else:
            return LLMError(
                error=f"Unsupported translation unit: {unit}",
                error_type="unsupported_unit",
            )

    @staticmethod
    def extract_generation(text: str) -> TextGenerationResponse:
        """Extract text generation response."""
        return ResponseValidator.validate_text_generation(text)

    @staticmethod
    def extract_response(text: str, expected_type: str = "auto") -> ParsedLLMResponse:
        """
        Extract and validate LLM response, auto-detecting type if needed.

        Args:
            text: Raw LLM response text
            expected_type: Expected response type ('word', 'sentence', 'text', 'auto')

        Returns:
            ParsedLLMResponse with validation metadata
        """
        validation_errors = []

        try:
            if expected_type == "auto":
                # Try to auto-detect response type
                if "translation" in text.lower() and "lemma" in text.lower():
                    parsed = ResponseValidator.validate_word_translation(text)
                elif "translation" in text.lower():
                    parsed = ResponseValidator.validate_sentence_translation(text)
                else:
                    parsed = ResponseValidator.validate_text_generation(text)
            elif expected_type == "word":
                parsed = ResponseValidator.validate_word_translation(text)
            elif expected_type == "sentence":
                parsed = ResponseValidator.validate_sentence_translation(text)
            elif expected_type == "text":
                parsed = ResponseValidator.validate_text_generation(text)
            else:
                parsed = ResponseValidator.validate_error_response(text)
                validation_errors.append(f"Unknown expected type: {expected_type}")

            return ParsedLLMResponse(
                raw_content=text,
                parsed_content=parsed,
                validation_errors=validation_errors,
            )

        except Exception as e:
            logger.error(f"Failed to extract LLM response: {e}")
            return ParsedLLMResponse(
                raw_content=text,
                parsed_content=ResponseValidator.validate_error_response(str(e)),
                validation_errors=[str(e)],
            )
