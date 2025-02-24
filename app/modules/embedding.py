from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_vectorstore(docs):
    """Creates a FAISS vectorstore from documents."""
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    return FAISS.from_documents(docs, embedding_model)
