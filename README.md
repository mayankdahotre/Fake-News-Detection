# 📰 Fake News Detection

This project detects whether a news article is **Fake** or **Real** using **Natural Language Processing (NLP)** techniques.  
It uses:
- **TF-IDF** for text vectorization  
- **Logistic Regression** for classification  
- **Streamlit** for a user-friendly web interface  

---

## 📂 Project Structure
```
fake_news_detection/
│
├── app.py              # Streamlit app for prediction
├── train_model.py      # Script to train and save model & vectorizer
├── merge_csv.py        # Script to combine true.csv & fake.csv into dataset.csv
├── fake_news.py        # Text cleaning helper functions
├── dataset.csv         # Combined dataset of fake and real news
├── model.pkl           # Saved trained model
├── vector.pkl          # Saved TF-IDF vectorizer
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

---

## 📊 Dataset

You can use:
- [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
- [ISOT Fake News Dataset](https://www.uvic.ca/engineering/ece/isot/datasets/index.php)

If you have `true.csv` and `fake.csv` files, merge them using the script below.

---

## 🔄 Merge Dataset Files

Create a script called `merge_csv.py`:

```python
import pandas as pd

# Load CSV files
true_df = pd.read_csv('true.csv')
fake_df = pd.read_csv('fake.csv')

# Add labels
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# If 'text' column not present, combine title & content
if 'text' not in true_df.columns:
    true_df['text'] = true_df['title'].astype(str) + " " + true_df['content'].astype(str)
if 'text' not in fake_df.columns:
    fake_df['text'] = fake_df['title'].astype(str) + " " + fake_df['content'].astype(str)

# Keep only required columns
true_df = true_df[['text', 'label']]
fake_df = fake_df[['text', 'label']]

# Combine and shuffle
df = pd.concat([true_df, fake_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# Save merged file
df.to_csv('dataset.csv', index=False)

print(f"✅ dataset.csv created with {len(df)} records.")
```

Run it:
```bash
python merge_csv.py
```

---

## ⚙️ Installation

1️⃣ Clone this repository:
```bash
git clone https://github.com/<your-username>/fake_news_detection.git
cd fake_news_detection
```

2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

Run:
```bash
python train_model.py
```
This will:
- Load `dataset.csv`
- Clean text
- Convert to TF-IDF features
- Train Logistic Regression
- Save the model as `model.pkl`
- Save the vectorizer as `vector.pkl`

---

## 🚀 Run the Web App

Run:
```bash
streamlit run app.py
```
Then open the provided **local URL** in your browser.

---

## 📦 Requirements

Contents of `requirements.txt`:
```
streamlit
pandas
scikit-learn
joblib
```
Install via:
```bash
pip install -r requirements.txt
```

---

## 🧠 Model Details

- **Feature Extraction:** TF-IDF (max_features=5000)
- **Classifier:** Logistic Regression (max_iter=200)
- **Metrics:** Accuracy, Precision, Recall, F1-score

---

## 🖥️ Example

**Input:**
```
The government announced new reforms to improve the economy.
```

**Output:**
```
✅ Real News
```

---

## 📜 License
MIT License
