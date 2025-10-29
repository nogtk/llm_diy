import re
from tokenizer import SimpleTokenizerV1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    verdict = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', verdict)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(f"語彙数: {vocab_size}語")

vocab = {token:integer for integer, token in enumerate(all_words)}

tokenizer = SimpleTokenizerV1(vocab)
text = """"
It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride.
"""
ids = tokenizer.encode(text)
print("エンコード結果:", ids)
print("デコード結果:", tokenizer.decode(ids))
