from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import re
import math
import json
from collections import Counter
from pypdf import PdfReader
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
DATABASE_URL = "sqlite:///./vectorflow.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DocumentModel(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    saved_path = Column(String)
    status = Column(String, default="PROCESSING")  # PROCESSING, COMPLETED, FAILED

class ChunkModel(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, index=True)
    text = Column(Text)
    vector = Column(Text)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI(title="VectorFlow API")
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

VOCABULARY = {}

def tokenize(text: str):
    return re.findall(r'\b\w+\b', text.lower())

def extract_and_chunk(file_path: str, chunk_size: int = 500, overlap: int = 50):
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + " "
            
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def build_vector(tokens, vocab):
    counts = Counter(tokens)
    vec = [counts.get(w, 0) for w in vocab]
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]

def cosine_similarity(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

# Background Worker Task
def process_document_pipeline(document_id: int, file_path: str):
    global VOCABULARY
    db = SessionLocal()
    try:
        chunks = extract_and_chunk(file_path)
        all_tokens = []
        chunk_tokens_list = []
        for c in chunks:
            toks = tokenize(c)
            chunk_tokens_list.append(toks)
            all_tokens.extend(toks)
            
        vocab_words = sorted(list(set(all_tokens)))
        VOCABULARY = {w: i for i, w in enumerate(vocab_words)}
        
        for toks, text_chunk in zip(chunk_tokens_list, chunks):
            vec = build_vector(toks, VOCABULARY)
            db_chunk = ChunkModel(
                document_id=document_id,
                text=text_chunk,
                vector=json.dumps(vec)
            )
            db.add(db_chunk)
            
        doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if doc:
            doc.status = "COMPLETED"
        db.commit()
    except Exception as e:
        doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
        if doc:
            doc.status = "FAILED"
        db.commit()
    finally:
        db.close()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class RAGRequest(BaseModel):
    question: str
    top_k: int = 3

@app.get("/")
def home():
    return {"message": "VectorFlow server is running successfully!"}

@app.post("/api/v1/upload")
def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    db_doc = DocumentModel(
        filename=file.filename,
        saved_path=file_path,
        status="PROCESSING"
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)

    # Directly run pipeline taaki chunks turant ready ho jayein
    process_document_pipeline(db_doc.id, file_path)

    return {
        "document_id": db_doc.id,
        "filename": file.filename,
        "status": "COMPLETED",
        "message": "File indexed successfully!"
    }

@app.get("/api/v1/documents/{document_id}/status")
def get_document_status(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    chunk_count = db.query(ChunkModel).filter(ChunkModel.document_id == document_id).count()
    return {
        "document_id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "total_chunks": chunk_count
    }

@app.post("/api/v1/search")
def search_similar(request: SearchRequest, db: Session = Depends(get_db)):
    if not VOCABULARY:
        return {"error": "Pehle document upload karke index hone ka wait karein!"}
        
    query_tokens = tokenize(request.query)
    query_vec = build_vector(query_tokens, VOCABULARY)
    
    all_chunks = db.query(ChunkModel).all()
    if not all_chunks:
        return {"error": "Database me koi chunks nahi hain!"}
        
    scored_results = []
    for item in all_chunks:
        doc_vec = json.loads(item.vector)
        if len(doc_vec) < len(query_vec):
            doc_vec.extend([0] * (len(query_vec) - doc_vec))
        elif len(query_vec) < len(doc_vec):
            query_vec.extend([0] * (len(doc_vec) - len(query_vec)))
            
        score = cosine_similarity(query_vec, doc_vec)
        scored_results.append((score, item.text, item.document_id))
        
    scored_results.sort(key=lambda x: x[0], reverse=True)
    top_results = scored_results[:request.top_k]
    
    return {
        "query": request.query,
        "results": [
            {"score": round(score, 4), "document_id": doc_id, "text": text}
            for score, text, doc_id in top_results
        ]
    }
@app.post("/api/v1/ask")
def ask_question_rag(request: RAGRequest, db: Session = Depends(get_db)):
    search_res = search_similar(SearchRequest(query=request.question, top_k=request.top_k), db=db)
    
    retrieved_chunks = []
    if "results" in search_res and search_res["results"]:
        positive_matches = [r["text"] for r in search_res["results"] if r.get("score", 0) > 0]
        retrieved_chunks = positive_matches if positive_matches else [r["text"] for r in search_res["results"][:2]]

    # Agar koi word match na ho, toh database se default chunks context me do
    if not retrieved_chunks:
        all_db_chunks = db.query(ChunkModel).limit(3).all()
        retrieved_chunks = [c.text for c in all_db_chunks]

    context_text = "\n\n---\n\n".join(retrieved_chunks)

    prompt = f"""You are VectorFlow AI Assistant. Answer the user's question clearly. 
If relevant information exists in the provided context, use it as primary reference.
Context:
{context_text}

User Question: {request.question}
Answer:"""

    try:
        model = genai.GenerativeModel("gemini-pro")
        res = model.generate_content(prompt)
        ai_answer = res.text
    except Exception as e:
        ai_answer = f"LLM Generation Error: {str(e)}"

    return {
        "question": request.question,
        "answer": ai_answer,
        "response": ai_answer,
        "sources_used": len(retrieved_chunks)
    }
# trigger deploy
