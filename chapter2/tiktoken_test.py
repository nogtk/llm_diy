# GPT は BPE (Byte Pair Encoding) を使ってトークン化を行っている
# tiktoken は BPEアルゴリズムを実装したライブラリ
# BPE は未知の単語を既知のサブワードに分割し、解釈する。なので |unk| のような未知の語彙に対するトークンを定義する必要がない

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terracesof someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)

## 練習問題2-1
text2 = "Akwirw ier"
integers2 = tokenizer.encode(text2)
print(integers2)

strings2 = tokenizer.decode(integers2)
print(strings2)
