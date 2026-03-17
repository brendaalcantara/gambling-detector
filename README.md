# Gambling Domain Classifier

A TFLite model for binary classification of gambling/betting domains vs. safe domains, using character n-grams and numeric features.

The classifier analyzes the **domain name** (e.g., `bet365.com`, `google.com`) and determines whether it belongs to a gambling site — useful for DNS filters, parental controls, browser extensions, or any application that needs to classify URLs.

## Approach

**Pipeline:**

1. Preprocessing — TLD removal, letter/digit separation
2. Feature extraction — character n-grams (2–4 chars) + numeric features (entropy, length, digit/vowel/special ratios)
3. Model — Conv1D + Dense layers with Focal Loss
4. Export — TFLite (~1–3 MB), optimized for on-device inference

## Project Structure

```
gambling-domain-classifier/
├── common.py                      # shared utilities (features, vocab, preprocessing)
├── train.py                       # training script (main pipeline)
├── evaluate.py                    # detailed model evaluation
├── data/
│   └── dataset_balanced.csv       # sample dataset (gambling + safe)
├── output/                        # generated artifacts (not versioned)
│   ├── gambling_classifier.tflite
│   ├── vocab.txt
│   └── labels.txt
├── notebook/
│   └── train_colab.ipynb          # alternative pipeline (tflite-model-maker)
├── tests/
│   └── test_common.py             # unit tests
├── requirements.txt
└── LICENSE
```

## Requirements

- **Python 3.9+**
- TensorFlow 2.15+
- No GPU required

## Usage

### 1. Train

```bash
# Default settings
python train.py

# Tune hyperparameters
python train.py --epochs 30 --batch 64

# Smaller vocab -> smaller model
python train.py --max-vocab 3000
```

### 2. Evaluate

```bash
# Full evaluation (requires trained model)
python evaluate.py

# Adjust classification threshold
python evaluate.py --threshold 0.7
```

### 3. Tests

```bash
pytest tests/
```

### Google Colab (alternative)

1. Open `notebook/train_colab.ipynb` in Google Colab
2. Run all cells in order
3. Upload `data/dataset_balanced.csv` when prompted
4. Download the generated `.tflite` model

> **Note:** The notebook uses `tflite-model-maker` (AverageWordVec), a simpler architecture than `train.py` (Conv1D + numeric features).

### Docker

```bash
docker run -it --rm \
  -v $(pwd):/workspace \
  -w /workspace \
  python:3.9-slim \
  bash -c "pip install -r requirements.txt && python train.py"
```

### Conda

```bash
conda create -n gambling-detector python=3.9 -y
conda activate gambling-detector
pip install -r requirements.txt
python train.py
```

## Targets

| Metric               | Target  |
|----------------------|---------|
| Accuracy             | >= 90%  |
| Precision (gambling) | >= 85%  |
| Recall (gambling)    | >= 80%  |
| Model size           | < 5 MB  |
| Latency (device)     | < 50ms  |

## Dataset

The repository includes a **sample dataset with 2,000 entries** (1,000 per class), enough to demonstrate the pipeline and train a functional model.

For better results, expand the dataset using the public sources below:

**Gambling sources:**
- [StevenBlack/hosts](https://github.com/StevenBlack/hosts) — consolidated blocklist
- [Blocklistproject](https://github.com/blocklistproject/Lists) — categorized lists
- [Sinfonietta](https://github.com/nicholasgasior/sinfonietta) — gambling blocklist
- Real and synthetic RDGA (Registered DGA) domains

**Safe sources:**
- [Tranco](https://tranco-list.eu/) — domain popularity ranking
- `.br` domains from Tranco 1M
- Curated whitelist of known services

## Using the TFLite Model

The exported model (`gambling_classifier.tflite`) can be integrated into any platform that supports TFLite:

- **Android/iOS** — via TFLite Interpreter
- **Python** — via `tf.lite.Interpreter`
- **Web** — via TFLite.js

Model inputs:
1. `ngram_input` — character n-gram index sequence (int32, shape `[1, 128]`)
2. `feat_input` — 5 normalized numeric features (float32, shape `[1, 5]`)

Output: probabilities `[gambling, safe]` (float32, shape `[1, 2]`)

See `common.py` for the reference implementation of `preprocess_domain`, `extract_char_ngrams`, `compute_features`, and `tokenize`.

## Intended Use & Ethical Guidelines

This project is designed for **protective and compliance purposes**:

- **Intended uses:** Parental controls, DNS filtering, corporate network policies, browser extensions that block or warn about gambling content, support for responsible gambling, and compliance in jurisdictions where online gambling is restricted.
- **Not intended for:** Facilitating access to gambling sites, marketing to gamblers, or any use that promotes or enables gambling.

The software is provided under the MIT license. The MIT license includes a **warranty disclaimer** and **limitation of liability** — the authors are not responsible for misuse or damages arising from the use of this software. Use at your own discretion and in accordance with applicable laws.

## License

[MIT](LICENSE)
