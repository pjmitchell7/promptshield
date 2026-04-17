from __future__ import annotations
import tiktoken

from collections import Counter

class TokenizerWrapper:
    # I am keeping the tokenizer logic isolated here so the rest of the project
    # does not have to care about the tiktoken API directly.
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)

    def encode(self, text: str) -> list[int]:
        # This turns raw prompt text into token IDs.
        return self.encoding.encode(text)

    def decode(self, tokens: list[int]) -> str:
        # This is mostly here for debugging or inspection later.
        return self.encoding.decode(tokens)