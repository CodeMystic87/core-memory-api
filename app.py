from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
import pinecone
import openai

# ----------------- Setup -----------------
app = FastAPI()

# Initialize OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV", "us-east-1")
index_name = os.getenv("INDEX_NAME", "core-memory")

pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
index = pinecone.Index(index_name)

# ----------------- Request Models -----------------
class MemoryRequest(BaseModel):
    text: str

class VocabularyRequest(BaseModel):
    words: List[str]

class SearchRequest(BaseModel):
    query: str
    topk: int = 5

# ----------------- Endpoints -----------------

@app.post("/storeMemory")
async def store_memory(req: MemoryRequest):
    """Store a single memory snippet"""
    embedding = openai.Embedding.create(
        model="text-embedding-3-small",
        input=req.text
    )["data"][0]["embedding"]

    index.upsert([(req.text, embedding, {"text": req.text})])
    return {"status": "stored", "memory": req.text}

@app.post("/storeVocabulary")
async def store_vocabulary(req: VocabularyRequest):
    """Store a whole list of vocabulary words"""
    for word in req.words:
        embedding = openai.Embedding.create(
            model="text-embedding-3-small",
            input=word
        )["data"][0]["embedding"]
        index.upsert([(word, embedding, {"text": word})])

    return {"status": "stored", "count": len(req.words)}

@app.post("/searchMemories")
async def search_memories(req: SearchRequest):
    """Search memories with semantic similarity"""
    embedding = openai.Embedding.create(
        model="text-embedding-3-small",
        input=req.query
    )["data"][0]["embedding"]

    results = index.query(vector=embedding, top_k=req.topk, include_metadata=True)
    matches = [
        {"text": match["metadata"]["text"], "score": match["score"]}
        for match in results["matches"]
    ]
    return {"results": matches}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
