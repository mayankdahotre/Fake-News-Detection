# Fake News Detection

This project implements two complete pipelines for detecting fake news articles:

1. **TF‑IDF + Logistic Regression** (fast, strong baseline)
2. **Transformer Fine‑Tuning (DistilBERT)** using Hugging Face

---

## Features
- End-to-end pipelines for text classification
- Robust data preprocessing
- Stratified train/val/test split
- Class imbalance handling
- Detailed metrics (accuracy, precision, recall, F1, ROC‑AUC)
- Model persistence for inference
- CLI to switch between models

---

## Requirements
```bash
# Baseline model dependencies
pip install pandas scikit-learn numpy matplotlib joblib

# Transformer model dependencies (GPU recommended)
pip install transformers datasets evaluate torch accelerate
```

---

## Dataset Format
CSV file with at least two columns: one for text and one for labels.
```csv
text,label
"Some news article...",fake
"Another news story...",real
```

---

## Usage
### Train with TF‑IDF + Logistic Regression
```bash
python fake_news.py \
    --data path/to/dataset.csv \
    --text_col text \
    --label_col label \
    --model tfidf_lr \
    --output_dir artifacts_tfidf
```

### Train with Transformer (DistilBERT)
```bash
python fake_news.py \
    --data path/to/dataset.csv \
    --text_col text \
    --label_col label \
    --model transformer \
    --epochs 3 \
    --batch_size 16 \
    --output_dir artifacts_bert
```

---

## Output
After training, the script saves:
- Model artifacts (`.joblib` for TF‑IDF, Hugging Face model files for Transformer)
- Label encoder
- Metrics in JSON format (`metrics.json`)
- Confusion matrix and classification report

---

## Inference
You can load the saved model and run predictions:
```python
from fake_news import infer_tfidf_lr, infer_transformer

# TF‑IDF model
results = infer_tfidf_lr(["Some text here"], "artifacts_tfidf")
print(results)

# Transformer model
results = infer_transformer(["Some text here"], "artifacts_bert")
print(results)
```

---

## Notes
- GPU is strongly recommended for the Transformer pipeline.
- Adjust hyperparameters (`epochs`, `batch_size`, `lr`) based on dataset size.
- Works for binary or multiclass classification.
