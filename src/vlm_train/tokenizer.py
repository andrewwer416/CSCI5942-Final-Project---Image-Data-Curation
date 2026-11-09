"""A small word-level tokenizer whose vocabulary comes only from train2017 captions."""

from __future__ import annotations

from collections import Counter
import re
from typing import Iterable


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def split_words(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


class Vocabulary:
    def __init__(self, words: list[str]) -> None:
        self.words = words
        self.ids = {word: index for index, word in enumerate(words)}
        self.pad_id = self.ids[PAD_TOKEN]
        self.bos_id = self.ids[BOS_TOKEN]
        self.eos_id = self.ids[EOS_TOKEN]
        self.unk_id = self.ids[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.words)

    @classmethod
    def from_texts(cls, texts: Iterable[str], min_frequency: int = 5) -> "Vocabulary":
        """Build a deterministic vocabulary, most frequent words first."""

        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(split_words(text))
        kept = sorted(
            (word for word, count in counts.items() if count >= min_frequency),
            key=lambda word: (-counts[word], word),
        )
        return cls([*SPECIAL_TOKENS, *kept])

    def encode(self, text: str, max_length: int = 77) -> list[int]:
        """Return exactly ``max_length`` ids: <bos>, words, <eos>, then padding."""

        if max_length < 2:
            raise ValueError("max_length must leave room for <bos> and <eos>")
        word_ids = [self.ids.get(word, self.unk_id) for word in split_words(text)]
        ids = [self.bos_id, *word_ids[: max_length - 2], self.eos_id]
        return ids + [self.pad_id] * (max_length - len(ids))
