import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_groq import ChatGroq
import pickle
from config import LLM_API_KEY

def get_answer(question, file_path):
    """Processes the question using the LLM and returns an answer with sources."""
    if not os.path.exists(file_path):
        return {"answer": "No processed data found.", "sources": ""}
    
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    llm = ChatGroq(api_key=LLM_API_KEY, model="llama-3.3-70b-versatile", temperature=0.9)
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    
    return chain({"question": question}, return_only_outputs=True)
