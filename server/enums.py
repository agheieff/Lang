from enum import Enum, IntFlag, auto


class RetryComponent(IntFlag):
    WORDS = 1
    SENTENCES = 2
    STRUCTURED = 4
