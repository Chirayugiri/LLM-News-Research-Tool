import os
import dotenv

dotenv.load_dotenv()

LLM_API_KEY = os.getenv("LLAMA_API_KEY")
FILE_PATH = "./vector_db/faiss_store_openai.pkl"
