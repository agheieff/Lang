from enum import Enum, IntFlag, auto

class TextUnit(str, Enum):
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    TEXT = "text"

class RetryComponent(IntFlag):
    WORDS = 1
    SENTENCES = 2
    STRUCTURED = 4
