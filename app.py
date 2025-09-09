from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
from openai import OpenAI
from pinecone import Pinecone
from datetime import datetime
import pytz

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("INDEX_NAME", "core-memory")
index = pc.Index(index_name)

# --------- Request Models ---------
class MemoryRequest(BaseModel):
    id: Optional[str] = None
    text: str
    tags: Optional[List[str]] = []
    kind: str = "note"  # default
    mood: Optional[str] = None
    people: Optional[List[str]] = []
    activities: Optional[List[str]] = []
    keywords: Optional[List[str]] = []
    meta: Optional[dict] = {}

class SearchRequest(BaseModel):
    query: Optional[str] = None
    date: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    kinds: Optional[List[str]] = None
    tags_contains: Optional[List[str]] = None
    tags_contains_any: Optional[List[str]] = None
    tags_contains_all: Optional[List[str]] = None
    people_contains_any: Optional[List[str]] = None
    mood_contains_any: Optional[List[str]] = None
    activities_contains_any: Optional[List[str]] = None
    include_text: bool = True
    sort_by: Optional[str] = "newest"
    summarize: bool = False

class UpdateRequest(BaseModel):
    id: str
    text: Optional[str] = None
    tags: Optional[List[str]] = None
    mood: Optional[str] = None
    people: Optional[List[str]] = None
    activities: Optional[List[str]] = None
    keywords: Optional[List[str]] = None

class DeleteRequest(BaseModel):
    id: str

class VocabularyRequest(BaseModel):
    words: List[str]

# --------- Helper: Ensure safe metadata ---------
def ensure_meta(meta: Optional[dict]) -> dict:
    """Always return a complete, safe metadata dictionary."""
    if not meta:
        meta = {}
    if "datetime_iso" not in meta or not meta["datetime_iso"]:
        meta["datetime_iso"] = datetime.now(pytz.utc).isoformat()
    if "timezone" not in meta or not meta["timezone"]:
        meta["timezone"] = "UTC"
    if "version" not in meta or not meta["version"]:
        meta["version"] = "v3.7a"
    return meta

# --------- Endpoints ---------
@app.post("/storeMemory")
def store_memory(request: MemoryRequest):
    request.meta = ensure_meta(request.meta)

    # Create embedding
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=request.text
    ).data[0].embedding

    # Use provided ID or fallback to datetime-based ID
    memory_id = request.id or request.meta["datetime_iso"]

    # Upsert into Pinecone
    index.upsert([{
        "id": memory_id,
        "values": embedding,
        "metadata": {
            "text": request.text,
            "tags": request.tags,
            "kind": request.kind,
            "mood": request.mood,
            "people": request.people,
            "activities": request.activities,
            "keywords": request.keywords,
            "meta": request.meta
        }
    }])

    return {
        "status": "ok",
        "memory": memory_id,
        "tags": request.tags,
    }

@app.post("/searchMemories")
def search_memories(request: SearchRequest):
    # Generate query embedding
    if request.query:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=request.query
        ).data[0].embedding
    else:
        embedding = [0.0] * 1536

    filters = {}
    if request.kinds:
        filters["kind"] = {"$in": request.kinds}
    if request.tags_contains_any:
        filters["tags"] = {"$in": request.tags_contains_any}

    results = index.query(
        vector=embedding,
        top_k=10,
        include_metadata=True,
        filter=filters if filters else None
    )

    return {"results": results}

@app.post("/updateMemory")
def update_memory(request: UpdateRequest):
    # fetch existing memory
    existing = index.fetch([request.id])
    if not existing or request.id not in existing["vectors"]:
        return {"status": "error", "message": "Memory not found"}

    metadata = existing["vectors"][request.id]["metadata"]

    # update fields
    if request.text is not None:
        metadata["text"] = request.text
    if request.tags is not None:
        metadata["tags"] = request.tags
    if request.mood is not None:
        metadata["mood"] = request.mood
    if request.people is not None:
        metadata["people"] = request.people
    if request.activities is not None:
        metadata["activities"] = request.activities
    if request.keywords is not None:
        metadata["keywords"] = request.keywords

    # regenerate embedding if text changed
    if request.text is not None:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=request.text
        ).data[0].embedding
    else:
        embedding = None

    index.upsert([{
        "id": request.id,
        "values": embedding if embedding else existing["vectors"][request.id]["values"],
        "metadata": metadata
    }])

    return {"status": "ok", "memory": request.id, "tags": metadata.get("tags", [])}

@app.post("/deleteMemory")
def delete_memory(request: DeleteRequest):
    index.delete(ids=[request.id])
    return {"status": "ok", "deleted_id": request.id}

@app.post("/storeVocabulary")
def store_vocabulary(request: VocabularyRequest):
    return {"status": "ok", "count": len(request.words)}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/testNoConfirm")
def test_no_confirm():
    return {"message": "API is live!"}
