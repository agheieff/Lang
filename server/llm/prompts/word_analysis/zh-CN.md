Now please analyze the following sentence: {sentence}

Return pipe-separated CSV format for each word in order:
word|translation|pos|pinyin

Note: For Chinese, the lemma is always the same as the word surface, so we omit it.

Tokenize into smallest meaningful units (words, not characters unless single-char words).
For words with multiple characters like "吃饭", tokenize as one word.
For compounds like "一个", tokenize as two words.
Skip punctuation.
Include all words in order, including duplicates.

IMPORTANT: Measure words (量词) must be translated as:
- "个" → "measure word"
- Other specific measure words → "measure word for [what they measure]"
  Examples: "辆" → "measure word for vehicles", "本" → "measure word for books",
           "只" → "measure word for animals", "张" → "measure word for flat objects"

Example for sentence "今天天气很好，阳光明媚。":
今天|today|NOUN|jīntiān
天气|weather|NOUN|tiānqì
很|very|ADV|hěn
好|good|ADJ|hǎo
阳光|sunshine|NOUN|yáng guāng
明媚|bright and beautiful|ADJ|míng mèi

Example for sentence "我要买一辆车。" with measure words:
我|I|PRON|wǒ
要|want|VERB|yào
买|buy|VERB|mǎi
一|one|NUM|yī
辆|measure word for vehicles|NOUN|liàng
车|car|NOUN|chē

Example for sentence "这个人有三个苹果。" with generic measure word:
这|this|DET|zhè
个|measure word|NOUN|gè
人|person|NOUN|rén
有|have|VERB|yǒu
三|three|NUM|sān
个|measure word|NOUN|gè
苹果|apple|NOUN|píng guǒ
