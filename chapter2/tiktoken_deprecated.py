# Deprecated local copy of tiktoken example
# This file was renamed from `tiktoken.py` because it shadowed the
# installed `tiktoken` package and caused AttributeError at runtime.

# GPT は BPE (Byte Pair Encoding) を使ってトークン化を行っている
# tiktoken は BPEアルゴリズムを実装したライブラリ

from importlib.metadata import version
import tiktoken_test

tokenizer = tiktoken_test.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

# Keep this file only for reference. Do not import it from project code.
