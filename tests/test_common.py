"""Unit tests for common.py."""

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (
    preprocess_domain,
    extract_char_ngrams,
    compute_features,
    build_vocab,
    tokenize,
    save_vocab,
    load_vocab,
    save_labels,
    texts_to_sequences,
    compute_all_features,
    NUM_FEATURES,
    SEQ_LEN,
)


class TestPreprocessDomain:
    def test_removes_http(self):
        assert preprocess_domain("http://google.com") == "google"

    def test_removes_https(self):
        assert preprocess_domain("https://google.com") == "google"

    def test_removes_www(self):
        assert preprocess_domain("www.google.com") == "google"

    def test_removes_simple_tld(self):
        assert preprocess_domain("google.com") == "google"

    def test_removes_compound_tld(self):
        assert preprocess_domain("google.com.br") == "google"

    def test_separates_letters_then_digits(self):
        assert preprocess_domain("bet365.com") == "bet 365"

    def test_separates_digits_then_letters(self):
        result = preprocess_domain("1xbet.com")
        assert "1" in result and "xbet" in result

    def test_lowercase(self):
        assert preprocess_domain("GOOGLE.COM") == "google"

    def test_strips_whitespace(self):
        assert preprocess_domain("  google.com  ") == "google"

    def test_removes_path(self):
        assert preprocess_domain("google.com/search?q=test") == "google"

    def test_short_name_fallback(self):
        result = preprocess_domain("ab.com")
        assert len(result) > 0

    def test_numeric_only_fallback(self):
        result = preprocess_domain("123.com")
        assert result == "123"

    def test_subdomain_takes_last_part(self):
        result = preprocess_domain("mail.google.com")
        assert result == "google"

    def test_bet_tld(self):
        result = preprocess_domain("garfo777.bet")
        assert "garfo" in result and "777" in result


class TestExtractCharNgrams:
    def test_bigrams(self):
        ngrams = extract_char_ngrams("abc")
        assert "ab" in ngrams
        assert "bc" in ngrams

    def test_trigrams(self):
        ngrams = extract_char_ngrams("abcd")
        assert "abc" in ngrams
        assert "bcd" in ngrams

    def test_fourgrams(self):
        ngrams = extract_char_ngrams("abcde")
        assert "abcd" in ngrams
        assert "bcde" in ngrams

    def test_ignores_spaces(self):
        ngrams = extract_char_ngrams("a b c d e")
        assert "ab" in ngrams

    def test_empty_string(self):
        assert extract_char_ngrams("") == []

    def test_single_char(self):
        assert extract_char_ngrams("a") == []

    def test_two_chars(self):
        ngrams = extract_char_ngrams("ab")
        assert ngrams == ["ab"]


class TestComputeFeatures:
    def test_output_shape(self):
        features = compute_features("google")
        assert features.shape == (NUM_FEATURES,)
        assert features.dtype == np.float32

    def test_empty_string_returns_zeros(self):
        features = compute_features("")
        assert np.all(features == 0)

    def test_all_digits_high_digit_ratio(self):
        features = compute_features("12345")
        digit_ratio = features[2]
        assert digit_ratio == pytest.approx(1.0)

    def test_no_digits_zero_digit_ratio(self):
        features = compute_features("google")
        digit_ratio = features[2]
        assert digit_ratio == pytest.approx(0.0)

    def test_values_normalized_between_0_and_1(self):
        features = compute_features("bet365gambling")
        assert all(0.0 <= f <= 1.0 for f in features)

    def test_high_entropy_random_string(self):
        features = compute_features("qjziwjsn")
        entropy = features[0]
        assert entropy > 0.3

    def test_low_entropy_repeated_string(self):
        features = compute_features("aaaa")
        entropy = features[0]
        assert entropy == pytest.approx(0.0)

    def test_length_normalization_capped_at_30(self):
        features_short = compute_features("abc")
        features_long = compute_features("a" * 50)
        assert features_long[1] == pytest.approx(1.0)
        assert features_short[1] < features_long[1]


class TestComputeAllFeatures:
    def test_batch_shape(self):
        result = compute_all_features(["google", "bet365", "abc"])
        assert result.shape == (3, NUM_FEATURES)
        assert result.dtype == np.float32


class TestBuildVocab:
    def test_includes_pad_and_unk(self):
        vocab = build_vocab(["abc", "def"])
        assert vocab["<PAD>"] == 0
        assert vocab["<UNK>"] == 1

    def test_max_vocab_limits_size(self):
        texts = [f"domain{i}abcdef" for i in range(100)]
        vocab = build_vocab(texts, max_vocab=10)
        assert len(vocab) <= 12  # 10 + PAD + UNK

    def test_most_frequent_first(self):
        vocab = build_vocab(["aaaa", "aabb", "aacc"], max_vocab=100)
        assert vocab.get("aa", 999) < vocab.get("bb", 999)

    def test_empty_input(self):
        vocab = build_vocab([])
        assert len(vocab) == 2  # PAD + UNK


class TestTokenize:
    def test_output_length(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "go": 2, "oo": 3}
        result = tokenize("google", vocab, max_len=10)
        assert len(result) == 10

    def test_pads_short_sequences(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "ab": 2}
        result = tokenize("ab", vocab, max_len=5)
        assert result[-1] == 0  # PAD

    def test_truncates_long_sequences(self):
        vocab = {"<PAD>": 0, "<UNK>": 1}
        result = tokenize("a" * 100, vocab, max_len=5)
        assert len(result) == 5

    def test_unknown_ngrams_get_unk_index(self):
        vocab = {"<PAD>": 0, "<UNK>": 1}
        result = tokenize("xyz", vocab, max_len=5)
        assert result[0] == 1  # UNK


class TestTextsToSequences:
    def test_output_shape(self):
        vocab = {"<PAD>": 0, "<UNK>": 1, "go": 2}
        result = texts_to_sequences(["google", "bet365"], vocab)
        assert result.shape == (2, SEQ_LEN)
        assert result.dtype == np.int32


class TestVocabIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        vocab = {"<PAD>": 0, "<UNK>": 1, "ab": 2, "bc": 3, "abc": 4}
        path = str(tmp_path / "vocab.txt")
        save_vocab(vocab, path)
        loaded = load_vocab(path)
        assert loaded == vocab

    def test_order_preserved(self, tmp_path):
        vocab = {"<PAD>": 0, "<UNK>": 1, "zz": 2, "aa": 3}
        path = str(tmp_path / "vocab.txt")
        save_vocab(vocab, path)
        loaded = load_vocab(path)
        assert loaded["zz"] == 2
        assert loaded["aa"] == 3


class TestSaveLabels:
    def test_saves_in_order(self, tmp_path):
        label_map = {"gambling": 0, "safe": 1}
        path = str(tmp_path / "labels.txt")
        save_labels(label_map, path)
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert lines == ["gambling", "safe"]
