# DocuMind – AI Powered PDF Chatbot

DocuMind is an AI-powered RAG (Retrieval-Augmented Generation) application that allows users to upload PDF documents and ask questions from them using natural language.

The system retrieves relevant information from the uploaded document and generates accurate answers using Large Language Models (LLMs).

---

# Features

- Upload PDF documents
- Extract text from PDFs
- Text chunking
- Vector embeddings
- Semantic similarity search
- AI-generated answers
- Human-friendly chatbot interface
- Context-aware responses
- RAG pipeline implementation

---

# Tech Stack


## Backend
- Python
- 
## AI / RAG
- ChromaDB
- Embedding Models
- Gemini / Groq API

---

# How It Works

```text
PDF Upload
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
Store in Vector Database
   ↓
User Question
   ↓
Similarity Search
   ↓
Relevant Context
   ↓
LLM Response
```

---

# Project Structure

```bash
DocuMind/
│
├── data/PDF
├── rag.py
├── requirements.txt
├── .env
├── README.md
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/ramvyasrnv/DocuMind.git
```

## Navigate to Project

```bash
cd DocuMind
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python rag.py
```

---

# Future Improvements

- Multi-PDF support
- Voice assistant integration
- AI agents
- Streaming responses

---

# Learning Outcomes

This project helped in understanding:

- RAG architecture
- Vector databases
- Embeddings
- Semantic search
- Prompt engineering
- LangChain workflow
- LLM integration

---

# Author

Ram Vyas

GitHub:
https://github.com/ramvyasrnv
