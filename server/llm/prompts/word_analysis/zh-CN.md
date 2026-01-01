Now please analyze the following sentence: {sentence}

Return pipe-separated CSV format for each word in order:
word|translation|pos|lemma|pinyin

Tokenize into smallest meaningful units (words, not characters unless single-char words).
For words with multiple characters like "吃饭", tokenize as one word.
For compounds like "一个", tokenize as two words.
Skip punctuation.
Include all words in order, including duplicates.

Example for sentence "今天天气很好，阳光明媚。":
今天|today|NOUN|今天|jīntiān
天气|weather|NOUN|天气|tiānqì
很|very|ADV|很|hěn
好|good|ADJ|好|hǎo
阳光|sunshine|NOUN|阳光|yáng guāng
明媚|bright and beautiful|ADJ|明媚|míng mèi
