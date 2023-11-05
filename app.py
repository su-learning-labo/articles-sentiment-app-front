import streamlit as st
import requests

st.title('ニュース記事の感情分析')

response = requests.get('https://articles-sentiment-app-back.onrender.com/articles/')
if response.status_code == 200:
    articles = response.json()
    for article in articles:
        st.write(f"Rank: {article['ranking']}")
        st.write(f"Title: {article['title']}")
        st.write(f"Description: {article['description']}")
        st.write(f"Sentiment: {article['sentiment']}, Score: {article['score']}")
        st.write("---")
else:
    st.write("Error fetching articles")