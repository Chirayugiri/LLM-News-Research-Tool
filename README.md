# 📰 LLM News Research Tool

An AI-powered web application that helps users gain deep insights from news articles using Large Language Models (LLMs). Built with **Streamlit** for frontend and **LangChain** + **custom modules** for processing and answering questions about news content.

---

## 🚨 Problem Statement

In today’s fast-paced digital age, it's challenging to:

- Read and understand multiple news articles quickly.
- Extract unbiased and factual insights from different sources.
- Ask specific questions about the contents of multiple articles.

**This project solves these challenges** by allowing users to input URLs of news articles, process them using NLP techniques, and interactively ask questions to get summarized, source-linked answers — all using LLMs.

---

## ✅ Features

- 🔗 Input multiple news article URLs (up to 10)
- 📄 Automatically fetch and preprocess content from articles
- 🧠 Vectorize data using Sentence Transformers and FAISS
- 💬 Ask natural language questions and get concise, intelligent answers
- 🔍 View referenced sources for transparency
- 💻 Clean and responsive UI using Streamlit

---

## 📦 Tech Stack

| Component       | Tool/Library              |
|----------------|---------------------------|
| Frontend       | Streamlit                 |
| LLM Interface  | LangChain, HuggingFace, Groq |
| Embeddings     | Sentence Transformers     |
| Vector Storage | FAISS                     |
| Content Parsing| Unstructured              |
| Hosting        | Render                    |

---

## 🚀 Demo

Live App: [🔗 Click here to try it out](https://llm-news-research-tool.onrender.com)

---

## 🔧 Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Chirayugiri/LLM-News-Research-Tool.git
   cd LLM-News-Research-Tool
