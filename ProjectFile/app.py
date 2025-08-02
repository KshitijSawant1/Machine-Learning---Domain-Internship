import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Preprocessing setup
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered = [t for t in tokens if t.isalnum()]
    filtered = [t for t in filtered if t not in stopwords.words('english') and t not in string.punctuation]
    stemmed = [ps.stem(t) for t in filtered]
    return " ".join(stemmed)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('models/vectorizer.pkl', 'rb'))
    model = pickle.load(open('models/model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Ensure 'vectorizer.pkl' and 'model.pkl' exist.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Spam Classifier", layout="centered")
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message", key="unique_text_input_1")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # Step 1: Preprocess
        transformed_sms = transform_text(input_sms)

        # Step 2: Vectorize (pass transformed!)
        vector_input = tfidf.transform([transformed_sms])

        # Step 3: Predict
        result = model.predict(vector_input)[0]

        # Step 4: Display
        st.subheader("Prediction:")
        if result == 1:
            st.error("Spam")
        else:
            st.success("Not Spam")
