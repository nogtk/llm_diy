import re
from tokenizer_v2 import SimpleTokenizerV2

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    verdict = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', verdict)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])

vocab = {token:integer for integer, token in enumerate(all_tokens)}

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of  the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.decode(tokenizer.encode(text)))
