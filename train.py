#!/usr/bin/env python3
"""
train.py — Train a TFLite model for gambling domain classification.

Architecture: Conv1D over character n-grams + numeric features
(entropy, length, digit/vowel/special ratios).

Input:  data/dataset_balanced.csv  (columns: text_column, label)
Output: output/gambling_classifier.tflite
        output/vocab.txt
        output/labels.txt

Usage:
    python train.py                          # defaults
    python train.py --epochs 30 --batch 64   # tune hyperparameters
    python train.py --data my_dataset.csv    # custom dataset
    python train.py --max-vocab 3000         # smaller vocab -> smaller model
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from common import (
    SEQ_LEN,
    EMBED_DIM,
    NUM_FEATURES,
    build_vocab,
    texts_to_sequences,
    compute_all_features,
    save_vocab,
    save_labels,
)


def load_dataset(csv_path):
    """Load and validate a CSV dataset."""
    texts, labels = [], []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames or "text_column" not in reader.fieldnames or "label" not in reader.fieldnames:
            print(f"ERROR: CSV must have columns 'text_column' and 'label'")
            print(f"       Found columns: {reader.fieldnames}")
            sys.exit(1)

        for row in reader:
            text = row["text_column"].lower().strip()
            label = row["label"].strip()
            if text and label:
                texts.append(text)
                labels.append(label)

    unique_labels = sorted(set(labels))
    if len(unique_labels) != 2:
        print(f"WARNING: Expected 2 classes, found {len(unique_labels)}: {unique_labels}")

    seen = set()
    duplicates = 0
    for t in texts:
        if t in seen:
            duplicates += 1
        seen.add(t)
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate domains in dataset")

    return texts, labels


def build_model(vocab_size, num_classes):
    import tensorflow as tf

    reg = tf.keras.regularizers.l2(1e-4)

    ngram_input = tf.keras.Input(shape=(SEQ_LEN,), dtype=tf.int32, name="ngram_input")
    x = tf.keras.layers.Embedding(vocab_size, EMBED_DIM)(ngram_input)
    x = tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)

    feat_input = tf.keras.Input(shape=(NUM_FEATURES,), dtype=tf.float32, name="feat_input")
    feat = tf.keras.layers.Dense(16, activation="relu")(feat_input)

    merged = tf.keras.layers.Concatenate()([x, feat])
    merged = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg)(merged)
    merged = tf.keras.layers.Dropout(0.3)(merged)
    merged = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=reg)(merged)
    merged = tf.keras.layers.Dropout(0.3)(merged)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(merged)

    model = tf.keras.Model(inputs=[ngram_input, feat_input], outputs=output)
    return model


def focal_loss(gamma=2.0, alpha=0.25):
    """Focal Loss — gives more weight to hard examples, reduces overconfidence."""
    import tensorflow as tf

    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_true_one_hot = tf.one_hot(tf.cast(tf.squeeze(y_true), tf.int32), depth=tf.shape(y_pred)[-1])
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        weight = alpha * y_true_one_hot * tf.pow(1.0 - y_pred, gamma)
        return tf.reduce_sum(weight * cross_entropy, axis=-1)

    return loss_fn


def export_tflite(model, output_path):
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return len(tflite_model)


def main():
    parser = argparse.ArgumentParser(description="Train TFLite model for gambling detection")
    parser.add_argument("--data", default="data/dataset_balanced.csv", help="Path to training CSV")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--split", type=float, default=0.9, help="Train fraction (rest = test)")
    parser.add_argument("--max-vocab", type=int, default=5000, help="Maximum n-gram vocabulary size")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--model-name", default="gambling_classifier.tflite", help="TFLite filename")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: Dataset not found: {args.data}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Gambling Detector — TFLite Training")
    print("  Architecture: Conv1D + Numeric Features + Focal Loss")
    print("=" * 60)
    print(f"  Dataset:    {args.data}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch:      {args.batch}")
    print(f"  Max Vocab:  {args.max_vocab}")
    print(f"  Seq Length: {SEQ_LEN}")
    print(f"  Split:      {args.split:.0%} train / {1 - args.split:.0%} test")
    print(f"  Output:     {args.output_dir}/{args.model_name}")
    print("=" * 60)

    print("\n[1/5] Importing TensorFlow...")
    t0 = time.time()
    import tensorflow as tf
    print(f"       TF {tf.__version__} OK ({time.time() - t0:.1f}s)")

    print("\n[2/5] Loading dataset...")
    t0 = time.time()
    texts, labels = load_dataset(args.data)

    unique_labels = sorted(set(labels))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in labels], dtype=np.int32)

    vocab = build_vocab(texts, max_vocab=args.max_vocab)
    X_seq = texts_to_sequences(texts, vocab)
    X_feat = compute_all_features(texts)

    n_train = int(len(X_seq) * args.split)
    indices = np.random.RandomState(42).permutation(len(X_seq))
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_seq_train, X_feat_train, y_train = X_seq[train_idx], X_feat[train_idx], y[train_idx]
    X_seq_test, X_feat_test, y_test = X_seq[test_idx], X_feat[test_idx], y[test_idx]

    print(f"       Total:    {len(X_seq)} samples")
    print(f"       Train:    {len(X_seq_train)} samples")
    print(f"       Test:     {len(X_seq_test)} samples")
    print(f"       Vocab:    {len(vocab)} tokens (max {args.max_vocab})")
    print(f"       Features: {NUM_FEATURES} (entropy, length, digits, vowels, special)")
    print(f"       Labels:   {unique_labels}")
    print(f"       OK ({time.time() - t0:.1f}s)")

    train_counts = Counter(y_train.tolist())
    total_train = len(y_train)
    n_classes = len(unique_labels)
    class_weight = {
        cls: total_train / (n_classes * count)
        for cls, count in train_counts.items()
    }
    print(f"       Class weights: {class_weight}")

    print(f"\n[3/5] Training model ({args.epochs} epochs)...")
    t0 = time.time()
    model = build_model(len(vocab), len(unique_labels))
    model.compile(
        optimizer="adam",
        loss=focal_loss(gamma=1.0, alpha=0.5),
        metrics=["accuracy"],
    )
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True,
    )
    model.fit(
        [X_seq_train, X_feat_train], y_train,
        epochs=args.epochs,
        batch_size=args.batch,
        validation_data=([X_seq_test, X_feat_test], y_test),
        callbacks=[early_stop],
        class_weight=class_weight,
        verbose=1,
    )
    train_time = time.time() - t0
    print(f"       Training completed in {train_time:.1f}s")

    print("\n[4/5] Evaluating...")
    loss, accuracy = model.evaluate([X_seq_test, X_feat_test], y_test, verbose=0)
    print(f"       Loss:     {loss:.4f}")
    print(f"       Accuracy: {accuracy:.2%}")

    print("\n[5/5] Exporting TFLite...")
    tflite_path = os.path.join(args.output_dir, args.model_name)
    model_bytes = export_tflite(model, tflite_path)
    size_mb = model_bytes / (1024 * 1024)
    print(f"       Model:  {tflite_path} ({size_mb:.2f} MB)")

    vocab_path = os.path.join(args.output_dir, "vocab.txt")
    labels_path = os.path.join(args.output_dir, "labels.txt")
    save_vocab(vocab, vocab_path)
    save_labels(label_map, labels_path)
    print(f"       Vocab:  {vocab_path} ({len(vocab)} tokens)")
    print(f"       Labels: {labels_path} ({unique_labels})")

    print("\n" + "=" * 60)
    print("  RESULT")
    print("=" * 60)
    print(f"  Accuracy:    {accuracy:.2%}  {'OK' if accuracy >= 0.90 else 'BELOW target (90%)'}")
    print(f"  Size:        {size_mb:.2f} MB  {'OK' if size_mb < 5 else 'ABOVE limit (5 MB)'}")
    print(f"  Train time:  {train_time:.1f}s")
    print(f"  Output:      {tflite_path}")
    print("=" * 60)

    if accuracy < 0.90 or size_mb >= 5:
        print("\n  Try adjusting: --epochs 50 or --batch 16")
        print()


if __name__ == "__main__":
    main()
