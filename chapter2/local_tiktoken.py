# Local copy of tiktoken usage for chapter2 examples.
# Renamed from tiktoken.py to avoid shadowing the installed 'tiktoken' package.
from importlib.metadata import version
import tiktoken_test

tokenizer = tiktoken_test.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
