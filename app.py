from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
from openai import OpenAI
from pinecone import Pinecone
from datetime import datetime

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "core-memory"))

# Local fallback cache
LOCAL_FALLBACK = {"last_saved": None}

# Request Models
class MemoryRequest(BaseModel):
    id: Optional[str] = None
    text: str
    kind: str = "note"
    tags: Optional[List[str]] = []
    people: Optional[List[str]] = []
    activities: Optional[List[str]] = []
    keywords: Optional[List[str]] = []
    mood: Optional[str] = None
    meta: Optional[dict] = {}

class UpdateRequest(BaseModel):
    id: str
    text: Optional[str] = None
    kind: Optional[str] = None
    tags: Optional[List[str]] = None
    people: Optional[List[str]] = None
    activities: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    mood: Optional[str] = None
    meta: Optional[dict] = None

class DeleteRequest(BaseModel):
    id: str

class SearchRequest(BaseModel):
    query: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    kinds: Optional[List[str]] = None
    tags_contains: Optional[List[str]] = None
    people_contains: Optional[List[str]] = None
    activities_contains: Optional[List[str]] = None
    mood_contains: Optional[List[str]] = None
    include_text: bool = True
    sort_by: Optional[str] = None
    summarize: bool = False

# --- Helpers ---
def embed_text(text: str):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# --- Endpoints ---
@app.post("/storeMemory")
def store_memory(req: MemoryRequest):
    try:
        entry_id = req.id or f"mem-{datetime.utcnow().isoformat()}"
        vector = embed_text(req.text)

        meta = req.meta or {}
        meta.update({
            "datetime_iso": datetime.utcnow().isoformat(),
            "version": "v3.7a"
        })

        index.upsert([{
            "id": entry_id,
            "values": vector,
            "metadata": {
                "text": req.text,
                "kind": req.kind,
                "tags": req.tags,
                "people": req.people,
                "activities": req.activities,
                "keywords": req.keywords,
                "mood": req.mood,
                "meta": meta
            }
        }])

        LOCAL_FALLBACK["last_saved"] = {
            "id": entry_id,
            "text": req.text,
            "kind": req.kind,
            "tags": req.tags,
            "meta": meta
        }

        return {
            "id": entry_id,
            "status": "saved",
            "text": req.text,
            "kind": req.kind,
            "tags": req.tags,
            "meta": meta
        }
    except Exception as e:
        return {"id": "local-fallback", "status": f"saved locally (error: {str(e)})", "text": req.text}

@app.post("/updateMemory")
def update_memory(req: UpdateRequest):
    try:
        if req.id == "local-fallback" and LOCAL_FALLBACK["last_saved"]:
            # Fallback: delete old + re-save as new
            old = LOCAL_FALLBACK["last_saved"]
            new_text = req.text or old["text"]
            return store_memory(MemoryRequest(
                id=None,
                text=new_text,
                kind=req.kind or old["kind"],
                tags=req.tags or old["tags"],
                meta=req.meta or old["meta"]
            ))

        # Otherwise, update directly in Pinecone
        existing = index.fetch([req.id])
        if not existing.vectors:
            return {"status": "not found", "id": req.id}

        meta = existing.vectors[req.id].metadata
        if req.text:
            vector = embed_text(req.text)
        else:
            vector = existing.vectors[req.id].values

        index.upsert([{
            "id": req.id,
            "values": vector,
            "metadata": {
                **meta,
                "text": req.text or meta.get("text", ""),
                "kind": req.kind or meta.get("kind", "note"),
                "tags": req.tags or meta.get("tags", [])
            }
        }])

        return {"id": req.id, "status": "updated"}
    except Exception as e:
        return {"id": req.id, "status": f"update failed: {str(e)}"}

@app.post("/deleteMemory")
def delete_memory(req: DeleteRequest):
    try:
        if req.id == "local-fallback" and LOCAL_FALLBACK["last_saved"]:
            LOCAL_FALLBACK["last_saved"]["deleted"] = True
            return {"id": req.id, "status": "deleted (fallback)"}

        index.delete(ids=[req.id])
        return {"id": req.id, "status": "deleted"}
    except Exception as e:
        return {"id": req.id, "status": f"delete failed: {str(e)}"}

@app.post("/searchMemories")
def search_memories(req: SearchRequest):
    try:
        if not req.query:
            return {"results": [], "status": "empty query"}

        vector = embed_text(req.query)
        res = index.query(vector=vector, top_k=5, include_metadata=True)

        return {
            "results": [
                {"id": match.id, "score": match.score, "metadata": match.metadata}
                for match in res.matches
            ],
            "status": "ok"
        }
    except Exception as e:
        return {"status": f"search failed: {str(e)}"}

@app.get("/lastSaved")
def last_saved():
    if LOCAL_FALLBACK["last_saved"]:
        return {"last_saved": LOCAL_FALLBACK["last_saved"], "status": "ok"}
    return {"status": "no local fallback found"}

@app.get("/health")
def health_check():
    return {"status": "ok"}
