from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import streamlit as st
import os
import pickle
import dotenv
import langchain
import hashlib
import torch

# Load environment variables
dotenv.load_dotenv()

# Set page configuration
st.set_page_config(page_title="News Research Tool", page_icon="📈", layout="wide")
langchain.debug = False  # Disable debug logs

# Model settings
model_path = "Model/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings = HuggingFaceEmbeddings(model_name=model_path, model_kwargs={"device": device})

# Cache directory
CACHE_DIR = "./vector_db"
os.makedirs(CACHE_DIR, exist_ok=True)

# UI Styling
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

# Sidebar inputs
st.sidebar.title("🔗 Input News Article URLs")
st.sidebar.markdown("Enter the URLs of the news articles you want to analyze.")
quantity = st.sidebar.number_input("Total URLs", min_value=1, max_value=10, value=1, step=1)
urls = [st.sidebar.text_input(f"URL {i+1}", key=f"url_input_{i}") for i in range(quantity)]
process_url_btn = st.sidebar.button("🚀 Process URLs", key="process_url_btn")

main_placeholder = st.empty()
file_path = os.path.join(CACHE_DIR, "combined_vectorstore.pkl")

# Create LLM instance
llm = ChatGroq(api_key=os.getenv("LLAMA_API_KEY"), model="llama-3.3-70b-versatile")

def hash_urls(url_list):
    """Create a hash of URL list for caching."""
    combined = "".join(url_list).encode("utf-8")
    return hashlib.md5(combined).hexdigest()

# Process URLs
if process_url_btn:
    if not any(urls):
        st.sidebar.error("❌ Please enter at least one valid URL.")
    else:
        with st.spinner("⏳ Processing URLs..."):
            try:
                url_hash = hash_urls(urls)
                cache_file = os.path.join(CACHE_DIR, f"{url_hash}.pkl")

                if os.path.exists(cache_file):
                    with open(cache_file, "rb") as f:
                        vectorstore = pickle.load(f)
                    st.sidebar.success("✅ Loaded from cache!")
                else:
                    main_placeholder.text("Data Loading...Started...✅✅✅")
                    loader = UnstructuredURLLoader(urls=urls)
                    data = loader.load()

                    # Deduplicate documents
                    all_docs = list({doc.page_content: doc for doc in data}.values())

                    # Split
                    main_placeholder.text("Text Splitter...Started...✅✅✅")
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        # chunk_overlap=50,
                        separators=["\n\n", "\n", " "]
                    )
                    docs = splitter.split_documents(all_docs)

                    # Embedding
                    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                    vectorstore = FAISS.from_documents(docs, embeddings)

                    # Cache result
                    with open(cache_file, "wb") as f:
                        pickle.dump(vectorstore, f)

                    st.sidebar.success("✅ URLs processed and cached!")

                # Also save to default file_path (optional)
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)

                main_placeholder.empty()

            except Exception as e:
                st.sidebar.error(f"⚠️ Error: {e}")

# Question input section
st.markdown("### 🔍 Ask a Question")
question = st.text_input("Enter your question about the news articles:", key="question_input")

# Display answer
if question:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        with st.spinner("🤖 Generating Answer..."):
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain.invoke({"question": question})

        st.markdown("## ✨ Answer")
        st.info(result["answer"])

        # Sources
        sources = result.get("sources", "")
        if sources:
            st.markdown("### 📌 Sources")
            for source in sources.split("\n"):
                st.write(f"🔗 {source}")
    else:
        st.error("⚠️ No processed data found. Please process URLs first.")
