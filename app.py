from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
from openai import OpenAI
from pinecone import Pinecone

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME", "core-memory"))

# Local fallback memory tracker
last_saved_entry = None

# Request Models
class MemoryRequest(BaseModel):
    text: str
    tags: Optional[List[str]] = []
    mood: Optional[str] = None
    people: Optional[List[str]] = []
    activities: Optional[List[str]] = []
    keywords: Optional[List[str]] = []
    kind: Optional[str] = "note"  # default type
    meta: Optional[dict] = {}

class VocabularyRequest(BaseModel):
    words: List[str]

class SearchRequest(BaseModel):
    query: Optional[str] = None  # allow empty queries
    topk: Optional[int] = 5
    tags_contains_any: Optional[List[str]] = []
    tags_contains_all: Optional[List[str]] = []
    mood_contains_any: Optional[List[str]] = []
    people_contains_any: Optional[List[str]] = []
    activities_contains_any: Optional[List[str]] = []
    include_text: Optional[bool] = True
    sort_by: Optional[str] = "newest"
    summarize: Optional[bool] = False

# ========== ROUTES ==========

@app.post("/storeMemory")
def store_memory(req: MemoryRequest):
    global last_saved_entry
    try:
        # Create embedding
        embedding = openai_client.embeddings.create(
            input=req.text,
            model="text-embedding-3-small"
        ).data[0].embedding

        # Store in Pinecone
        index.upsert([{
            "id": req.meta.get("datetime_iso", req.text[:50]),
            "values": embedding,
            "metadata": req.dict()
        }])

        # Save fallback locally
        last_saved_entry = req.dict()

        return {"status": "ok", "memory": req.dict()}

    except Exception as e:
        # API fallback mode
        last_saved_entry = req.dict()
        return {"status": "local-only", "error": str(e), "memory": req.dict()}


@app.post("/searchMemories")
def search_memories(req: SearchRequest):
    try:
        # If no query, return last saved entry
        if not req.query:
            if last_saved_entry:
                return {
                    "results": [last_saved_entry],
                    "summary": "⚠️ API query skipped. Returning last locally saved entry."
                }
            return {"results": [], "summary": "No query provided and no local memory available."}

        # Generate embedding for query
        embedding = openai_client.embeddings.create(
            input=req.query,
            model="text-embedding-3-small"
        ).data[0].embedding

        # Query Pinecone
        res = index.query(vector=embedding, top_k=req.topk, include_metadata=True)

        return {"results": res.get("matches", []), "summary": None}

    except Exception as e:
        # Fallback if Pinecone fails
        if last_saved_entry:
            return {
                "results": [last_saved_entry],
                "summary": f"⚠️ Pinecone unavailable. Returning last locally saved entry. Error: {str(e)}"
            }
        return {"results": [], "summary": f"❌ Error: {str(e)}"}


@app.post("/updateMemory")
def update_memory(id: str, req: MemoryRequest):
    try:
        # Just overwrite metadata in Pinecone
        index.update(id=id, set_metadata=req.dict())
        return {"status": "ok", "updated": id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/deleteMemory")
def delete_memory(id: str):
    try:
        index.delete(ids=[id])
        return {"status": "ok", "deleted_id": id}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/storeVocabulary")
def store_vocabulary(req: VocabularyRequest):
    try:
        for word in req.words:
            embedding = openai_client.embeddings.create(
                input=word,
                model="text-embedding-3-small"
            ).data[0].embedding
            index.upsert([{
                "id": f"vocab-{word}",
                "values": embedding,
                "metadata": {"type": "vocab", "word": word}
            }])
        return {"status": "ok", "count": len(req.words)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/testNoConfirm")
def test_no_confirm():
    return {"message": "Hello from testNoConfirm!"}
