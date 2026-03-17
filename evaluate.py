#!/usr/bin/env python3
"""
evaluate.py — Detailed evaluation of a trained TFLite model.

Loads the .tflite model + vocab.txt, runs real inference,
and computes precision/recall/F1 per class.

Includes gray-zone (RDGA) domain tests.

Usage:
    python evaluate.py
    python evaluate.py --model output/gambling_classifier.tflite
    python evaluate.py --threshold 0.7   # only classify as gambling if >= 70%
"""

import argparse
import csv
import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

from common import (
    NUM_FEATURES,
    preprocess_domain,
    tokenize,
    compute_features,
    load_vocab,
)

GRAY_ZONE_GAMBLING = [
    ("82kkh2.com", "alphanumeric RDGA"),
    ("n0x2m5q.com", "alternating RDGA"),
    ("c3x7m0k.com", "alternating RDGA"),
    ("qjziwjsn.com", "random letters RDGA"),
    ("w4m8x1k.com", "alternating RDGA"),
    ("e4q7rst.com", "alternating RDGA"),
    ("o5c4pw39.com", "alphanumeric RDGA"),
    ("vp6gg5.com", "short RDGA"),
    ("h3q5c7m.com", "alternating RDGA"),
    ("y3m6x9k.com", "alternating RDGA"),
    ("d2x4b6q.com", "alternating RDGA"),
    ("e9z4q6v4.com", "alphanumeric RDGA"),
    ("t9y2w3o9.com", "alphanumeric RDGA"),
    ("377bet19.com", "numbers + bet"),
    ("we-ducatipg.com", "brand cloaking"),
    ("d4k8r9m6.com", "alternating RDGA"),
    ("ant-aa.co", "short cloaking"),
    ("u7pg-news.com", "cloaking + news"),
    ("garfo777.bet", "word + numbers"),
]

KNOWN_SAFE = [
    ("google.com", "global tech"),
    ("youtube.com", "streaming"),
    ("mercadolivre.com.br", "BR e-commerce"),
    ("nubank.com.br", "BR bank"),
    ("globo.com", "BR media"),
    ("instagram.com", "social media"),
    ("gov.br", "government"),
    ("uol.com.br", "BR portal"),
    ("itau.com.br", "BR bank"),
    ("amazon.com.br", "e-commerce"),
    ("wikipedia.org", "encyclopedia"),
    ("stackoverflow.com", "tech"),
    ("correios.com.br", "postal service"),
    ("ifood.com.br", "BR delivery"),
    ("spotify.com", "streaming"),
]

KNOWN_GAMBLING = [
    ("bet365.com", "global betting"),
    ("betano.com", "BR betting"),
    ("sportingbet.com", "sports betting"),
    ("betnacional.com", "BR betting"),
    ("pixbet.com", "BR betting"),
    ("brazino777.com", "BR casino"),
    ("blaze.com", "BR casino"),
    ("stake.com", "global casino"),
    ("f12.bet", "BR betting"),
    ("estrelabet.com", "BR betting"),
    ("betfair.com", "global betting"),
    ("888casino.com", "global casino"),
    ("pokerstars.com", "poker"),
    ("parimatch.com", "global betting"),
    ("1xbet.com", "global betting"),
]


def classify(interpreter, ngram_idx, feat_idx, output_idx, tokens, features,
             gambling_idx=0, threshold=0.5):
    """Run TFLite inference and return (label_index, confidence).

    Classifies as gambling if P(gambling) >= threshold.
    Tensor indices are resolved once by the caller, not per-call.
    """
    interpreter.set_tensor(ngram_idx, tokens.reshape(1, -1))
    interpreter.set_tensor(feat_idx, features.reshape(1, -1))
    interpreter.invoke()
    probs = interpreter.get_tensor(output_idx)[0]

    gambling_prob = float(probs[gambling_idx])
    if gambling_prob >= threshold:
        return gambling_idx, gambling_prob
    else:
        safe_idx = 1 - gambling_idx
        return safe_idx, float(probs[safe_idx])


def main():
    parser = argparse.ArgumentParser(description="Detailed TFLite model evaluation")
    parser.add_argument("--model", default="output/gambling_classifier.tflite")
    parser.add_argument("--vocab", default="output/vocab.txt")
    parser.add_argument("--labels", default="output/labels.txt")
    parser.add_argument("--data", default="data/dataset_balanced.csv")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Min P(gambling) to classify as gambling (default: 0.5)")
    args = parser.parse_args()

    for f in [args.model, args.vocab, args.labels]:
        if not os.path.exists(f):
            print(f"ERROR: File not found: {f}")
            print("       Run 'python train.py' first to generate model artifacts.")
            sys.exit(1)

    print("=" * 70)
    print("  Gambling Detector — Detailed TFLite Model Evaluation")
    print("=" * 70)

    with open(args.labels) as f:
        labels = [line.strip() for line in f if line.strip()]
    print(f"\n  Labels: {labels}")

    gambling_idx = labels.index("gambling") if "gambling" in labels else 0

    vocab = load_vocab(args.vocab)
    print(f"  Vocabulary: {len(vocab)} tokens")

    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    ngram_detail = None
    feat_detail = None
    for detail in input_details:
        if detail["shape"][-1] == NUM_FEATURES:
            feat_detail = detail
        else:
            ngram_detail = detail

    max_len = int(ngram_detail["shape"][-1])
    ngram_idx = ngram_detail["index"]
    feat_idx = feat_detail["index"]
    output_idx = output_details[0]["index"]

    size_mb = os.path.getsize(args.model) / (1024 * 1024)
    print(f"  Model: {args.model} ({size_mb:.2f} MB)")
    print(f"  Threshold: {args.threshold:.0%}")
    print(f"  Inputs: {len(input_details)} (ngrams: {ngram_detail['shape']}, features: {feat_detail['shape']})")
    print(f"  Output shape: {output_details[0]['shape']}")

    # =============================================
    # TEST 1: Gray-zone domains (RDGA)
    # =============================================
    print("\n" + "=" * 70)
    print("  TEST 1: Gray Zone — RDGA Domains (expected: gambling)")
    print("=" * 70)

    gray_correct = 0
    gray_total = len(GRAY_ZONE_GAMBLING)

    for domain, pattern in GRAY_ZONE_GAMBLING:
        name = preprocess_domain(domain)
        tokens = tokenize(name, vocab, max_len)
        features = compute_features(name)
        pred_idx, conf = classify(interpreter, ngram_idx, feat_idx, output_idx,
                                  tokens, features, gambling_idx, args.threshold)
        pred_label = labels[pred_idx]
        is_correct = pred_label == "gambling"
        if is_correct:
            gray_correct += 1
        status = "OK" if is_correct else "MISS"
        print(f"  [{status:4s}] {domain:25s} -> {pred_label:8s} ({conf:.1%})  [{name}] ({pattern})")

    print(f"\n  Gray zone: {gray_correct}/{gray_total} correct ({gray_correct/gray_total:.0%})")

    # =============================================
    # TEST 2: Known safe sites
    # =============================================
    print("\n" + "=" * 70)
    print("  TEST 2: Known Safe Sites (expected: safe)")
    print("=" * 70)

    safe_correct = 0
    safe_total = len(KNOWN_SAFE)

    for domain, desc in KNOWN_SAFE:
        name = preprocess_domain(domain)
        tokens = tokenize(name, vocab, max_len)
        features = compute_features(name)
        pred_idx, conf = classify(interpreter, ngram_idx, feat_idx, output_idx,
                                  tokens, features, gambling_idx, args.threshold)
        pred_label = labels[pred_idx]
        is_correct = pred_label == "safe"
        if is_correct:
            safe_correct += 1
        status = "OK" if is_correct else "FP"
        print(f"  [{status:4s}] {domain:25s} -> {pred_label:8s} ({conf:.1%})  [{name}] ({desc})")

    print(f"\n  Safe: {safe_correct}/{safe_total} correct ({safe_correct/safe_total:.0%})")

    # =============================================
    # TEST 3: Known gambling sites
    # =============================================
    print("\n" + "=" * 70)
    print("  TEST 3: Known Gambling Sites (expected: gambling)")
    print("=" * 70)

    gambl_correct = 0
    gambl_total = len(KNOWN_GAMBLING)

    for domain, desc in KNOWN_GAMBLING:
        name = preprocess_domain(domain)
        tokens = tokenize(name, vocab, max_len)
        features = compute_features(name)
        pred_idx, conf = classify(interpreter, ngram_idx, feat_idx, output_idx,
                                  tokens, features, gambling_idx, args.threshold)
        pred_label = labels[pred_idx]
        is_correct = pred_label == "gambling"
        if is_correct:
            gambl_correct += 1
        status = "OK" if is_correct else "MISS"
        print(f"  [{status:4s}] {domain:25s} -> {pred_label:8s} ({conf:.1%})  [{name}] ({desc})")

    print(f"\n  Gambling: {gambl_correct}/{gambl_total} correct ({gambl_correct/gambl_total:.0%})")

    # =============================================
    # TEST 4: Full dataset — Precision / Recall / F1
    # =============================================
    if not os.path.exists(args.data):
        print(f"\n  Dataset not found: {args.data}")
        return

    print("\n" + "=" * 70)
    print("  TEST 4: Batch Evaluation — Precision / Recall / F1")
    print("=" * 70)

    with open(args.data, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    tp = fp = tn = fn = 0
    errors_gambling = []
    errors_safe = []

    t0 = time.time()
    for row in rows:
        text = row["text_column"]
        true_label = row["label"]
        tok = tokenize(text, vocab, max_len)
        feat = compute_features(text)
        pred_idx, conf = classify(interpreter, ngram_idx, feat_idx, output_idx,
                                  tok, feat, gambling_idx, args.threshold)
        pred_label = labels[pred_idx]

        if true_label == "gambling" and pred_label == "gambling":
            tp += 1
        elif true_label == "safe" and pred_label == "gambling":
            fp += 1
            if len(errors_safe) < 10:
                errors_safe.append((text, conf))
        elif true_label == "safe" and pred_label == "safe":
            tn += 1
        elif true_label == "gambling" and pred_label == "safe":
            fn += 1
            if len(errors_gambling) < 10:
                errors_gambling.append((text, conf))

    eval_time = time.time() - t0
    total = tp + fp + tn + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    print(f"\n  Confusion Matrix:")
    print(f"                  Pred gambling   Pred safe")
    print(f"  True gambling   {tp:>8d} (TP)   {fn:>8d} (FN)")
    print(f"  True safe       {fp:>8d} (FP)   {tn:>8d} (TN)")

    print(f"\n  Metrics (gambling class):")
    prec_status = "OK" if precision >= 0.95 else ("CLOSE" if precision >= 0.85 else "LOW")
    rec_status = "OK" if recall >= 0.90 else ("CLOSE" if recall >= 0.80 else "LOW")
    f1_status = "OK" if f1 >= 0.92 else ("CLOSE" if f1 >= 0.85 else "LOW")

    print(f"  Precision:  {precision:.2%}  (target >= 95%)  [{prec_status}]")
    print(f"  Recall:     {recall:.2%}  (target >= 90%)  [{rec_status}]")
    print(f"  F1-Score:   {f1:.2%}  (target >= 92%)  [{f1_status}]")
    print(f"  Accuracy:   {accuracy:.2%}")
    print(f"  Time:       {eval_time:.1f}s ({total} samples, {total/eval_time:.0f}/s)")
    print(f"  Latency:    {eval_time/total*1000:.2f}ms per inference")

    if errors_gambling:
        print(f"\n  False negatives (gambling classified as safe) — top {len(errors_gambling)}:")
        for text, conf in errors_gambling:
            print(f"    [{text}] conf={conf:.1%}")

    if errors_safe:
        print(f"\n  False positives (safe classified as gambling) — top {len(errors_safe)}:")
        for text, conf in errors_safe:
            print(f"    [{text}] conf={conf:.1%}")

    # =============================================
    # SUMMARY
    # =============================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Model:      {args.model} ({size_mb:.2f} MB)")
    print(f"  Vocabulary: {len(vocab)} tokens")
    print(f"  Features:   {NUM_FEATURES} numeric")
    print(f"  Threshold:  {args.threshold:.0%}")
    print(f"  Gray zone:  {gray_correct}/{gray_total} ({gray_correct/gray_total:.0%})")
    print(f"  Safe:       {safe_correct}/{safe_total} ({safe_correct/safe_total:.0%})")
    print(f"  Gambling:   {gambl_correct}/{gambl_total} ({gambl_correct/gambl_total:.0%})")
    print(f"  Precision:  {precision:.2%}")
    print(f"  Recall:     {recall:.2%}")
    print(f"  F1-Score:   {f1:.2%}")
    print(f"  Accuracy:   {accuracy:.2%}")

    all_ok = precision >= 0.85 and recall >= 0.80 and f1 >= 0.85
    print(f"\n  Verdict: {'PASSED' if all_ok else 'NEEDS IMPROVEMENT'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
