"""
common.py — Shared utilities for gambling domain classification.

Contains feature extraction, n-gram processing, preprocessing,
and vocabulary I/O used by both train.py and evaluate.py.
"""

import math
import re
from collections import Counter

import numpy as np

SEQ_LEN = 128
EMBED_DIM = 32
NGRAM_RANGE = (2, 4)
NUM_FEATURES = 5

_COMPOUND_TLD = re.compile(r"\.(com|net|org|gov|edu|mil)\.[a-z]{2}$")
_SIMPLE_TLD = re.compile(r"\.[a-z]{2,}$")
_LETTERS_DIGITS = re.compile(r"([a-z]+)(\d+)")
_DIGITS_LETTERS = re.compile(r"(\d+)([a-z]+)")


def preprocess_domain(domain: str) -> str:
    """Normalize a raw domain into the format expected by the model.

    Strips protocol, www prefix, TLD, and separates letters from digits.
    Example: "www.bet365.com" -> "bet 365"
    """
    domain = domain.lower().strip()
    domain = re.sub(r"^(https?://)", "", domain)
    domain = re.sub(r"^www\.", "", domain)
    domain = domain.split("/")[0]

    without_tld = _COMPOUND_TLD.sub("", domain)
    without_tld = _SIMPLE_TLD.sub("", without_tld)

    parts = without_tld.split(".")
    name = parts[-1] if parts else domain

    if not name or len(name) <= 2 or name.isdigit():
        return domain.split(".")[0]

    result = _LETTERS_DIGITS.sub(r"\1 \2", name)
    result = _DIGITS_LETTERS.sub(r"\1 \2", result)
    return result.strip()


def extract_char_ngrams(text: str) -> list:
    """Extract character n-grams (2 to 4) from text, ignoring spaces."""
    text = text.replace(" ", "")
    ngrams = []
    for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
        for i in range(len(text) - n + 1):
            ngrams.append(text[i : i + n])
    return ngrams


def build_vocab(texts: list, max_vocab: int = 5000) -> dict:
    """Build n-gram vocabulary from training texts."""
    counter = Counter()
    for text in texts:
        counter.update(extract_char_ngrams(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ngram, _ in counter.most_common(max_vocab):
        vocab[ngram] = len(vocab)
    return vocab


def texts_to_sequences(texts: list, vocab: dict) -> np.ndarray:
    """Convert a list of texts into a matrix of n-gram indices (batch)."""
    seqs = np.zeros((len(texts), SEQ_LEN), dtype=np.int32)
    for i, text in enumerate(texts):
        ngrams = extract_char_ngrams(text)
        for j, ng in enumerate(ngrams[:SEQ_LEN]):
            seqs[i, j] = vocab.get(ng, 1)
    return seqs


def tokenize(text: str, vocab: dict, max_len: int) -> np.ndarray:
    """Convert a single text into a sequence of indices (single-sample inference).

    Uses max_len derived from the loaded model rather than the SEQ_LEN constant,
    ensuring compatibility with models trained with different sequence lengths.
    """
    ngrams = extract_char_ngrams(text)
    unk_idx = vocab.get("<UNK>", 1)
    pad_idx = vocab.get("<PAD>", 0)
    indices = [vocab.get(ng, unk_idx) for ng in ngrams]

    if len(indices) < max_len:
        indices.extend([pad_idx] * (max_len - len(indices)))
    else:
        indices = indices[:max_len]

    return np.array(indices, dtype=np.int32)


def compute_features(text: str) -> np.ndarray:
    """Extract 5 normalized numeric features from a domain name.

    Returns [entropy, normalized_length, digit_ratio, vowel_ratio, special_ratio].
    All values are in the [0, 1] range.
    """
    raw = text.replace(" ", "")
    length = len(raw)

    if length == 0:
        return np.zeros(NUM_FEATURES, dtype=np.float32)

    freq = Counter(raw)
    probs = [c / length for c in freq.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)

    digit_ratio = sum(1 for c in raw if c.isdigit()) / length
    vowel_ratio = sum(1 for c in raw if c in "aeiou") / length
    special_ratio = sum(1 for c in raw if c in "-_.") / length
    norm_length = min(length, 30) / 30.0

    return np.array(
        [entropy / 5.0, norm_length, digit_ratio, vowel_ratio, special_ratio],
        dtype=np.float32,
    )


def compute_all_features(texts: list) -> np.ndarray:
    """Extract numeric features for a list of texts."""
    return np.array([compute_features(t) for t in texts], dtype=np.float32)


def save_vocab(vocab: dict, path: str) -> None:
    """Save vocabulary to a text file (one token per line, sorted by index)."""
    with open(path, "w", encoding="utf-8") as f:
        for word in sorted(vocab, key=vocab.get):
            f.write(f"{word}\n")


def load_vocab(vocab_path: str) -> dict:
    """Load vocabulary from a text file."""
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            token = line.strip()
            if token:
                vocab[token] = idx
    return vocab


def save_labels(label_map: dict, path: str) -> None:
    """Save label map to a text file (sorted by index)."""
    with open(path, "w", encoding="utf-8") as f:
        for label in sorted(label_map, key=label_map.get):
            f.write(f"{label}\n")
