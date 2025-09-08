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
index_name = os.getenv("INDEX_NAME", "core-memory")
index = pc.Index(index_name)

# ---------- Request Models ----------

class StoreMemoryRequest(BaseModel):
    text: str
    tags: Optional[List[str]] = []
    kind: Optional[str] = "note"
    title: Optional[str] = None
    mood: Optional[str] = None
    people: Optional[List[str]] = []
    activities: Optional[List[str]] = []
    keywords: Optional[List[str]] = []
    meta: Optional[dict] = {}

class UpdateMemoryRequest(BaseModel):
    id: str
    text: Optional[str] = None
    tags: Optional[List[str]] = []
    mood: Optional[str] = None
    people: Optional[List[str]] = []
    activities: Optional[List[str]] = []
    keywords: Optional[List[str]] = []
    meta: Optional[dict] = {}

class DeleteMemoryRequest(BaseModel):
    id: str

class VocabularyRequest(BaseModel):
    words: List[str]

class SearchRequest(BaseModel):
    query: Optional[str] = None  # ✅ Optional, defaults to None
    topk: int = 10
    date: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    kinds: Optional[List[str]] = []
    tags: Optional[List[str]] = []
    tags_contains_any: Optional[List[str]] = []
    tags_contains_all: Optional[List[str]] = []
    people_contains_any: Optional[List[str]] = []
    mood_contains_any: Optional[List[str]] = []
    activities_contains_any: Optional[List[str]] = []
    include_text: bool = True
    sort_by: str = "newest"
    summarize: bool = False

# ---------- Endpoints ----------

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/storeMemory")
async def store_memory(req: StoreMemoryRequest):
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.text
    ).data[0].embedding

    mem_id = f"mem-{hash(req.text)}"

    index.upsert([(
        mem_id,
        embedding,
        {"text": req.text, "tags": req.tags, "kind": req.kind, "title": req.title,
         "mood": req.mood, "people": req.people, "activities": req.activities,
         "keywords": req.keywords, "meta": req.meta}
    )])

    return {"status": "stored", "memory": mem_id, "tags": req.tags}

@app.post("/updateMemory")
async def update_memory(req: UpdateMemoryRequest):
    # Re-embed if text is updated
    embedding = None
    if req.text:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=req.text
        ).data[0].embedding

    index.upsert([(
        req.id,
        embedding if embedding else [0.0] * 1536,  # fallback vector
        {"text": req.text, "tags": req.tags, "mood": req.mood,
         "people": req.people, "activities": req.activities,
         "keywords": req.keywords, "meta": req.meta}
    )])

    return {"status": "updated", "memory": req.id, "tags": req.tags}

@app.post("/deleteMemory")
async def delete_memory(req: DeleteMemoryRequest):
    index.delete(ids=[req.id])
    return {"status": "deleted", "deleted_id": req.id}

@app.post("/searchMemories")
async def search_memories(req: SearchRequest):
    embedding = None

    # Case 1: Query exists → real embedding
    if req.query:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=req.query
        ).data[0].embedding
        print("DEBUG - Using real embedding for query:", req.query)
    else:
        # Case 2: No query → dummy vector
        embedding = [0.0] * 1536
        print("DEBUG - No query provided. Using dummy embedding.")

    # Build filters
    pinecone_filter = None
    if req.tags:
        pinecone_filter = {"tags": {"$in": req.tags}}
        print("DEBUG - Pinecone filter:", pinecone_filter)

    # Query Pinecone
    results = index.query(
        vector=embedding,
        top_k=req.topk,
        include_metadata=True,
        filter=pinecone_filter
    )
    print("DEBUG - Pinecone returned", len(results["matches"]), "matches")

    matches = [
        {
            "id": match["id"],
            "score": match["score"],
            "metadata": match["metadata"]
        }
        for match in results["matches"]
    ]

    return {"results": matches}
