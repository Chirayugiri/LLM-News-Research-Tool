import streamlit as st
import os
import pickle
import dotenv
import langchain
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load environment variables
dotenv.load_dotenv()

# Enable Langchain debugging
langchain.debug = False  # Turn off debugging for a cleaner UI

# Set page configuration
st.set_page_config(page_title="News Research Tool", page_icon="📈", layout="wide")

# Custom styling
st.markdown("""
    <style>
        .stTextInput>div>div>input {
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button {
            border-radius: 8px;
            background-color: #007BFF;
            color: white;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>News Research Tool 📰🔍</h1>", unsafe_allow_html=True)
st.write("Gain insights from news articles using AI-powered research.")

# Sidebar
st.sidebar.title("🔗 Input News Article URLs")
st.sidebar.markdown("Enter the URLs of the news articles you want to analyze.")

quantity = st.sidebar.number_input("Total URLs", min_value=1, max_value=10, value=1, step=1)
urls = [st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i}") for i in range(quantity)]

process_url_btn = st.sidebar.button("🚀 Process URLs", key="process_url_btn")

file_path = "./vector_db/faiss_store_openai.pkl"
main_placeholder = st.empty()

# Create LLM instance
llm = ChatGroq(api_key=os.getenv("LLAMA_API_KEY"), model="llama-3.3-70b-versatile", temperature=0.9)

# Process URLs
if process_url_btn:
    if not any(urls):
        st.sidebar.error("❌ Please enter at least one valid URL.")
    else:
        with st.spinner("⏳ Processing URLs... This may take a moment."):
            try:
                main_placeholder.text("Data Loading...Started...✅✅✅")
                loader = UnstructuredURLLoader(urls=urls)
                data = loader.load()

                # Split data
                splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=1000)
                main_placeholder.text("Text Splitter...Started...✅✅✅")
                docs = splitter.split_documents(data)

                # Embedding data
                main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
                vectorstore = FAISS.from_documents(docs, embedding_model)

                main_placeholder.empty()

                # Save vector data
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)

                st.sidebar.success("✅ URLs processed successfully!")
            except Exception as e:
                st.sidebar.error(f"⚠️ Error: {e}")

# Question input section
st.markdown("### 🔍 Ask a Question")
question = st.text_input("Enter your question about the news articles:", key="question_input")

# Display answer if a question is asked
if question:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
        
        with st.spinner("🤖 Generating Answer..."):
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": question}, return_only_outputs=True)

        st.markdown("## ✨ Answer")
        st.info(result["answer"])

        # Display sources
        sources = result.get("sources", "")
        if sources:
            st.markdown("### 📌 Sources")
            for source in sources.split("\n"):
                st.write(f"🔗 {source}")
    else:
        st.error("⚠️ No processed data found. Please process URLs first.")
