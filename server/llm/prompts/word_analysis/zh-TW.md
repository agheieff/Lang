Now please analyze the following sentence: {sentence}

Return pipe-separated CSV format for each word in order:
word|translation|pos|pinyin

Note: For Chinese, the lemma is always the same as the word surface, so we omit it.

Tokenize into smallest meaningful units (words, not characters unless single-char words).
For words with multiple characters like "吃飯", tokenize as one word.
For compounds like "一個", tokenize as two words.
Skip punctuation.
Include all words in order, including duplicates.

IMPORTANT: Measure words (量詞) must be translated as:
- "個" → "measure word"
- Other specific measure words → "measure word for [what they measure]"
  Examples: "輛" → "measure word for vehicles", "本" → "measure word for books",
           "隻" → "measure word for animals", "張" → "measure word for flat objects"

Example for sentence "今天天氣很好，陽光明媚。":
今天|today|NOUN|jīntiān
天氣|weather|NOUN|tiānqì
很|very|ADV|hěn
好|good|ADJ|hǎo
陽光|sunshine|NOUN|yáng guāng
明媚|bright and beautiful|ADJ|míng mèi

Example for sentence "我要買一輛車。" with measure words:
我|I|PRON|wǒ
要|want|VERB|yào
買|buy|VERB|mǎi
一|one|NUM|yī
輛|measure word for vehicles|NOUN|liàng
車|car|NOUN|chē

Example for sentence "這個人有三個蘋果。" with generic measure word:
這|this|DET|zhè
個|measure word|NOUN|gè
人|person|NOUN|rén
有|have|VERB|yǒu
三|three|NUM|sān
個|measure word|NOUN|gè
蘋果|apple|NOUN|píng guǒ
