import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Fake News Detector")

news = st.text_area("Enter News Text")

if st.button("Check News"):

    if news.strip() == "":
        st.warning("Please enter some news text")
    else:
        data = vectorizer.transform([news])
        prediction = model.predict(data)

        if prediction[0] == 0:
            st.error("Fake News")
        else:
            st.success("Real News")