Now please translate the following sentence from the text into English: {sentence}. Go over all the words from the sentence in order, not skipping anything, and return valid json - a single array of all the words. Each word should be an object with the following four keys:
- word – the original Chinese character(s)
- pinyin – the pinyin with diacritics (e.g. "tiānqì")
- translation – the most context-appropriate English gloss for this word
- pos – the part-of-speech label (e.g. "NOUN", "VERB", "ADJ", "PART", etc.)

Tokenize the text into the smallest meaningful units (words, not individual characters unless they are single-character words).
Choose the forms that can be found in a dictionary, for example "一个" should be tokenized as two, not as one, but something like "吃饭" should be tokenized as one word.
Preserve the original order; the array must map one-to-one with the token sequence. Skip any punctuation you encounter.
If you encounter the same word twice in the sentence, please put it twice in the response as well, the response should cover all the words in the sentence in their order, including duplicated ones.

Here's an example for how the first sentence from the text from before would be done:

[
    {
        "word": "今天",
        "pinyin": "jīntiān",
        "translation": "today",
        "pos": "NOUN"
    },
    {
        "word": "天气",
        "pinyin": "tiānqì",
        "translation": "weather",
        "pos": "NOUN"
    },
    {
        "word": "很",
        "pinyin": "hěn",
        "translation": "very",
        "pos": "ADV"
    },
    {
        "word": "好",
        "pinyin": "hǎo",
        "translation": "good",
        "pos": "ADJ"
    },
    {
        "word": "阳光",
        "pinyin": "yáng guāng",
        "translation": "sunshine",
        "pos": "NOUN"
    },
    {
        "word": "明媚",
        "pinyin": "míng mèi",
        "translation": "bright and beautiful",
        "pos": "ADJ"
    },
]
