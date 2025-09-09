from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import uuid
from openai import OpenAI
from pinecone import Pinecone
from fastapi.openapi.utils import get_openapi

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI + Pinecone
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("INDEX_NAME", "core-memory"))

# Global fallback cache
last_saved_entry = None

# -------------------------
# Models
# -------------------------

class MemoryRequest(BaseModel):
    text: str
    kind: Optional[str] = "note"
    tags: Optional[List[str]] = []
    mood: Optional[str] = None
    people: Optional[List[str]] = []
    activities: Optional[List[str]] = []
    keywords: Optional[List[str]] = []
    meta: Optional[Dict] = {}

class VocabularyRequest(BaseModel):
    words: List[str]

class SearchRequest(BaseModel):
    query: Optional[str] = None
    topk: Optional[int] = 5
    tags_contains_any: Optional[List[str]] = []
    tags_contains_all: Optional[List[str]] = []
    mood_contains_any: Optional[List[str]] = []
    people_contains_any: Optional[List[str]] = []
    activities_contains_any: Optional[List[str]] = []
    include_text: Optional[bool] = True
    sort_by: Optional[str] = "newest"
    summarize: Optional[bool] = False

# -------------------------
# Endpoints
# -------------------------

@app.post("/storeMemory")
def store_memory(req: MemoryRequest):
    global last_saved_entry
    try:
        embedding = openai_client.embeddings.create(
            input=req.text,
            model="text-embedding-3-small"
        ).data[0].embedding

        mem_id = str(uuid.uuid4())
        metadata = req.dict()

        index.upsert([{
            "id": mem_id,
            "values": embedding,
            "metadata": metadata
        }])

        last_saved_entry = {"id": mem_id, **metadata}

        return {"status": "ok", "memory": last_saved_entry}

    except Exception as e:
        last_saved_entry = {"id": "local-fallback", **req.dict()}
        return {"status": "fallback", "memory": last_saved_entry, "error": str(e)}


@app.post("/searchMemories")
def search_memories(req: SearchRequest):
    try:
        if not req.query:
            if last_saved_entry:
                return {
                    "results": [last_saved_entry],
                    "summary": "⚠️ No query provided. Returning last locally saved entry."
                }
            return {"results": [], "summary": "No query provided and no local memory available."}

        embedding = openai_client.embeddings.create(
            input=req.query,
            model="text-embedding-3-small"
        ).data[0].embedding

        res = index.query(vector=embedding, top_k=req.topk, include_metadata=True)

        return {"results": res.get("matches", []), "summary": None}

    except Exception as e:
        if last_saved_entry:
            return {
                "results": [last_saved_entry],
                "summary": f"⚠️ Pinecone unavailable. Returning last locally saved entry. Error: {str(e)}"
            }
        return {"results": [], "summary": f"❌ Error: {str(e)}"}


@app.post("/updateMemory")
def update_memory(id: str, req: MemoryRequest):
    try:
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


@app.get("/lastSaved")
def get_last_saved():
    if last_saved_entry:
        return {"status": "ok", "memory": last_saved_entry}
    else:
        return {"status": "empty", "message": "No memory has been saved yet."}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/testNoConfirm")
def test_no_confirm():
    return {"message": "Hello from testNoConfirm!"}

# -------------------------
# Custom OpenAPI Schema
# -------------------------

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="CoreMemory API",
        version="1.0.0",
        description=(
            "Custom memory storage + retrieval API with local fallback.\n\n"
            "⚠️ If Pinecone is unavailable, endpoints may return the last locally saved entry."
        ),
        routes=app.routes,
    )

    openapi_schema["info"]["x-fallback"] = (
        "If backend services fail, API will serve the last locally saved entry."
    )

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
