# import streamlit as st
# import joblib
# from fake_news import clean_text

# # Load model & vectorizer
# model = joblib.load('model.pkl')
# vectorizer = joblib.load('vectorizer.pkl')

# st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# st.title("📰 Fake News Detection App")
# st.write("Enter a news article and I’ll tell you if it's likely Fake or Real.")

# # Input box
# user_input = st.text_area("News content:")

# if st.button("Predict"):
#     if user_input.strip():
#         cleaned_text = clean_text(user_input)
#         input_vector = vectorizer.transform([cleaned_text])
#         prediction = model.predict(input_vector)[0]
#         st.subheader("Prediction:")
#         if prediction.lower() == "real":
#             st.success("✅ Real News")
#         else:
#             st.error("❌ Fake News")
#     else:
#         st.warning("Please enter some text.")


import streamlit as st
import joblib
import re
import string

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vector.pkl')

st.set_page_config("Fake News Detector", page_icon="📰")

st.title("Fake News Detection App")
st.write("Enter a news article below, and the model will predict whether it is Fake or Real.")

user_input = st.text_area("News Content")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    if prediction.lower() == "real":
        st.success("✅ Real News")
    else:
        st.error("❌ Fake News")
