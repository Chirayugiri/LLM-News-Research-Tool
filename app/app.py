import streamlit as st
import os
import pickle
from modules import llm, preprocessing, embedding
from config import LLM_API_KEY, FILE_PATH  
from assets.styles import apply_custom_styles
 
# Set up Streamlit page
st.set_page_config(page_title="News Research Tool", page_icon="📈", layout="wide")
apply_custom_styles()

# Title 
st.markdown("<h1>News Research Tool 📰</h1>", unsafe_allow_html=True)
st.write("Gain insights from news articles using AI-powered research.")

# Sidebar input
st.sidebar.title("🔗 Input News Article URLs")
st.sidebar.markdown("Enter the URLs of the news articles you want to analyze.")

quantity = st.sidebar.number_input("Total URLs", min_value=1, max_value=10, value=1, step=1)
urls = [st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i}") for i in range(quantity)]
process_url_btn = st.sidebar.button("🚀 Process URLs", key="process_url_btn")

# Process URLs
if process_url_btn:
    if not any(urls):
        st.sidebar.error("❌ Please enter at least one valid URL.")
    else:
        with st.spinner("⏳ Processing URLs... This may take a moment."):
            success, error_msg = preprocessing.process_urls(urls, FILE_PATH)
            if success:
                st.sidebar.success("✅ URLs processed successfully!")
            else:
                st.sidebar.error(f"⚠️ Error: {error_msg}")

# Question input
st.markdown("### 🔍 Ask a Question")
question = st.text_input("Enter your question about the news articles:", key="question_input")

# Generate answer
if question:
    if os.path.exists(FILE_PATH):
        with st.spinner("🤖 Generating Answer..."):
            result = llm.get_answer(question, FILE_PATH)
        
        st.markdown("## ✨ Answer")
        st.info(result["answer"])

        if result.get("sources"):
            st.markdown("### 📌 Sources")
            for source in result["sources"].split("\n"):
                st.write(f"🔗 {source}")
    else:
        st.error("⚠️ No processed data found. Please process URLs first.")
