import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load model
model = load_model("sentiment_model_gru_chatgpt.keras")

# Load tokenizer
with open("tokenizer_gru_chatgpt.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder (not used now, but kept for consistency)
with open("label_encoder_gru_chatgpt.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Set max length
max_len = 100

# Emoji map
emoji_map = {
    "Positive": "😊",
    "Negative": "😞",
    "Neutral": "😐"
}

# Custom index-to-label mapping
index_to_label = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# Cleaning
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# Predict
def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    pred = model.predict(padded)
    index = int(np.argmax(pred))
    label = index_to_label.get(index, "Unknown")
    confidence = float(pred[0][index])
    return label, confidence, emoji_map.get(label, "🧐")

# ChatGPT-inspired styling
st.markdown("""
<style>
    html, body, [class*="css"] {
        background-color: #1f1f1f;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextInput > div > div > input, .stTextArea textarea {
        background-color: #2d2d2d;
        color: #ffffff;
        border-radius: 10px;
        padding: 0.75rem;
    }
    .stButton > button {
        background-color: #0A9396;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #94D2BD;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# Show ChatGPT Logo
st.image("https://upload.wikimedia.org/wikipedia/commons/0/04/ChatGPT_logo.svg", width=80)

# App title
st.title("🌟 How was your ChatGPT experience today?")
st.markdown("#### Paste your review or share your thoughts — let AI read your mind! 🤖💬")

# User input
user_review = st.text_area("📝 Your Review:", placeholder="E.g. I absolutely love how ChatGPT helps with my coding doubts!")

# Submit button
if st.button("✨ Analyze Feedback"):
    if not user_review.strip():
        st.warning("Please write or paste your review first.")
    else:
        sentiment, confidence, emoji = predict_sentiment(user_review)

        # Output
        st.markdown("### 🧠 AI Response :")
        st.success(f"**Sentiment :** {sentiment} {emoji}")
        st.markdown(f"**Confidence Score :** {confidence:.2f}")
        st.progress(confidence)

# Footer
st.markdown("---")
st.caption("🚀 Built with ❤️ by Gayathri G Murali | GRU Neural Network | ChatGPT Reviews Analyzer")
