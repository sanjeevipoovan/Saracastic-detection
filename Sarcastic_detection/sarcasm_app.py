import streamlit as st
import pandas as pd
import tweepy
import re
from transformers import pipeline
import matplotlib.pyplot as plt

# -----------------------------
# Load sarcasm detection model
# -----------------------------
sarcasm_model = pipeline("text-classification", model="mrm8488/t5-base-finetuned-sarcasm-twitter")

# -----------------------------
# Preprocess tweet text
# -----------------------------
def preprocess(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return text.lower()

# -----------------------------
# Twitter API setup
# -----------------------------
def twitter_client():
    return tweepy.Client(
        bearer_token="AAAAAAAAAAAAAAAAAAAAAINl0wEAAAAAioHPqliUNl%2Fnz4a9ixl7v4Auv90%3DWDnlK4SG6ET3kk1HMca2nwUzYdLRTcZCHsk1P48C5FkfxTS8xP"
    )

# -----------------------------
# Fetch tweets from Twitter API
# -----------------------------
def fetch_tweets(query, count):
    client = twitter_client()
    tweets = []
    response = client.search_recent_tweets(query=query, tweet_fields=['text'], max_results=min(count, 100))
    if response.data:
        tweets = [tweet.text for tweet in response.data]
    return tweets

# -----------------------------
# Label Mapping
# -----------------------------
label_map = {
    "LABEL_0": "Not Sarcastic",
    "LABEL_1": "Sarcastic"
}
emoji_map = {
    "Not Sarcastic": "🙂",
    "Sarcastic": "😏"
}

# -----------------------------
# Streamlit App Interface
# -----------------------------
st.title("🤖 SarcoFizz ")

tab1, tab2 = st.tabs(["🔍 Twitter Sarcasm", "💬 User Input"])

# ----------- TAB 1: Twitter -----------
with tab1:
    st.subheader("Search Tweets and Detect Sarcasm")
    query = st.text_input("Enter a Twitter search query:", "CocaCola")
    count = st.slider("Number of tweets to fetch", 10, 100, 20)

    if st.button("Fetch and Analyze Tweets"):
        raw_tweets = fetch_tweets(f"{query} -is:retweet lang:en", count)
        cleaned = [preprocess(t) for t in raw_tweets]
        labels = [sarcasm_model(t)[0]['label'] for t in cleaned]
        readable_labels = [label_map.get(label, label) for label in labels]

        df = pd.DataFrame({'Tweet': raw_tweets, 'Sarcasm': readable_labels})
        st.write(df)

        df.to_csv("twitter_sarcasm_results.csv", index=False)
        st.success("✅ Results saved to twitter_sarcasm_results.csv")

        # Plot Sarcasm Distribution
        st.subheader("Sarcasm Distribution")
        st.bar_chart(df['Sarcasm'].value_counts())

# ----------- TAB 2: User Input -----------
with tab2:
    st.subheader("Check if your sentence is sarcastic")
    user_text = st.text_area("Enter text here:")

    if st.button("Analyze Text"):
        clean_text = preprocess(user_text)
        result = sarcasm_model(clean_text)[0]
        readable_label = label_map.get(result['label'], result['label'])
        emoji = emoji_map.get(readable_label, "")
        st.markdown(f"**Prediction:** {readable_label} {emoji} (Confidence: {result['score']:.2f})")
