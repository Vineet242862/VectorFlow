# ⚡ VectorFlow — Intelligent RAG Search Engine

VectorFlow is an asynchronous Retrieval-Augmented Generation (RAG) system built with FastAPI, SQLite, and Google Gemini. It ingests PDF documents, splits content into context chunks, retrieves relevant sections using TF-IDF ranking, and synthesizes accurate answers with loop control explanations and citations.

---

## 🚀 Features

- **Asynchronous Ingestion:** Processes PDF uploads in the background via FastAPI `BackgroundTasks`.
- **Custom Text Retrieval:** Tokenizes and scores document chunks with custom TF-IDF relevance ranking.
- **LLM Synthesis:** Augments query prompts with top-K context chunks for Google Gemini (`gemini-2.5-flash`).
- **Modern Dark UI:** Single-page responsive interface with live indexing feedback and structured markdown rendering.

---

## 🛠️ Tech Stack

- **Backend:** FastAPI, Python, SQLite, SQLAlchemy
- **Document Processing:** PyPDF
- **Model:** Google Generative AI SDK (`google-generativeai`)
- **Frontend:** HTML5, TailwindCSS, JavaScript

---

## 📦 Getting Started

### 1. Clone & Set Up

```bash
git clone [https://github.com/Vineet242862/VectorFlow.git](https://github.com/Vineet242862/VectorFlow.git)
cd VectorFlow
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
