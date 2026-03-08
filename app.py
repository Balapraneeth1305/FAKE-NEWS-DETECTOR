import pickle
import streamlit as st

# Load the trained model
with open("model/fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the TF-IDF vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

st.title("Fake News Detection System")

news_text = st.text_area("Paste the news article here:")

if st.button("Predict"):
    if news_text.strip() == "":
        st.warning("Please paste a news article.")
    else:
        transformed_text = tfidf.transform([news_text])
        prediction = model.predict(transformed_text)

        if prediction[0] == 1:
            st.success("✅ This news is REAL")
        else:
            st.error("❌ This news is FAKE")