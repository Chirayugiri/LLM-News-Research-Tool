from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os

# Set your local model path
LOCAL_MODEL_PATH = os.path.abspath("../Model/all-MiniLM-L6-v2")

def create_vectorstore(docs):
    """Creates a FAISS vectorstore from documents."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name=LOCAL_MODEL_PATH,
        model_kwargs={"device": device}
    )
    return FAISS.from_documents(docs, embedding_model)
