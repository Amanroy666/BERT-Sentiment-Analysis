"""Streamlit dashboard for sentiment analysis"""
import streamlit as st
import requests

st.title("🎭 BERT Sentiment Analysis")

text_input = st.text_area("Enter text for sentiment analysis:", height=150)

if st.button("Analyze Sentiment"):
    if text_input:
        response = requests.post(
            "http://localhost:8000/predict",
            json={"text": text_input}
        )
        
        result = response.json()
        
        st.subheader("Results:")
        st.write(f"**Sentiment:** {result['sentiment']}")
        st.write(f"**Confidence:** {result['confidence']:.2%}")
        
        st.bar_chart(result['probabilities'])
