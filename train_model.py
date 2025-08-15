# import pandas as pd
# import re
# import string
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report

# # Import helper function
# from fake_news import clean_text

# # Load dataset
# df = pd.read_csv('dataset.csv')

# # Preprocess
# df['text'] = df['text'].apply(clean_text)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     df['text'], df['label'], test_size=0.2, random_state=42
# )

# # Vectorization
# vectorizer = TfidfVectorizer(max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # Train model
# model = LogisticRegression(max_iter=200)
# model.fit(X_train_tfidf, y_train)

# # Evaluation
# y_pred = model.predict(X_test_tfidf)
# print(classification_report(y_test, y_pred))

# # Save model and vectorizer
# joblib.dump(model, 'model.pkl')
# joblib.dump(vectorizer, 'vectorizer.pkl')

# print("✅ Model and vectorizer saved successfully.")


# train_model.py

import pandas as pd
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load data
df = pd.read_csv('dataset.csv')
df['text'] = df['text'].apply(clean_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# Evaluation
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vector.pkl')
print("Model and vectorizer saved.")
