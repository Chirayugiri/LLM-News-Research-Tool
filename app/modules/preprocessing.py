import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from modules.embedding import create_vectorstore
import streamlit as st

def process_urls(urls, file_path):
    """Loads and processes news articles from URLs."""
    markdown_label = st.empty()
    try:
        markdown_label.text("Data Loading...Started...✅✅✅")
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        markdown_label.text("Text Splitter...Started...✅✅✅")
        splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=1000)
        docs = splitter.split_documents(data)
         
        markdown_label.text("Embedding Vector Started Building...✅✅✅")                           
        vectorstore = create_vectorstore(docs)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        return True, None
    except Exception as e:
        return False, str(e)
