from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
from openai import OpenAI
from pinecone import Pinecone
import uuid

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI + Pinecone
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

# -------------------------
# Models
# -------------------------

class MemoryRequest(BaseModel):
    text: str
    kind: str = "note"
    tags: Optional[List[str]] = []
    metadata: Optional[Dict] = {}

class SearchRequest(BaseModel):
    query: Optional[str] = None
    topk: int = 5
    tags_contains: Optional[List[str]] = []
    summarize: bool = False

class UpdateRequest(BaseModel):
    id: str
    text: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict] = None

class DeleteRequest(BaseModel):
    id: str

class VocabularyRequest(BaseModel):
    words: List[str]

# -------------------------
# Endpoints
# -------------------------

@app.post("/storeMemory")
def store_memory(req: MemoryRequest):
    embedding = openai_client.embeddings.create(
        input=req.text,
        model="text-embedding-3-small"
    ).data[0].embedding

    mem_id = str(uuid.uuid4())
    metadata = req.metadata or {}
    metadata.update({"kind": req.kind, "tags": req.tags})

    index.upsert([{
        "id": mem_id,
        "values": embedding,
        "metadata": metadata
    }])

    return {"status": "ok", "memory": req.text, "id": mem_id}


@app.post("/searchMemories")
def search_memories(req: SearchRequest):
    query_vec = None
    if req.query:
        query_vec = openai_client.embeddings.create(
            input=req.query,
            model="text-embedding-3-small"
        ).data[0].embedding
    else:
        return {"error": "No query provided"}

    res = index.query(vector=query_vec, top_k=req.topk, include_metadata=True)

    results = []
    for match in res.matches:
        results.append({
            "id": match.id,
            "score": match.score,
            "metadata": match.metadata
        })

    return {"results": results}


@app.post("/updateMemory")
def update_memory(req: UpdateRequest):
    # Fetch current memory
    res = index.fetch(ids=[req.id])
    if req.id not in res.vectors:
        return {"error": "Memory not found"}

    current = res.vectors[req.id]
    new_text = req.text if req.text else current["metadata"].get("text", "")
    embedding = openai_client.embeddings.create(
        input=new_text,
        model="text-embedding-3-small"
    ).data[0].embedding

    metadata = current["metadata"]
    if req.tags is not None:
        metadata["tags"] = req.tags
    if req.metadata is not None:
        metadata.update(req.metadata)

    index.upsert([{
        "id": req.id,
        "values": embedding,
        "metadata": metadata
    }])

    return {"status": "updated", "id": req.id}


@app.post("/deleteMemory")
def delete_memory(req: DeleteRequest):
    index.delete(ids=[req.id])
    return {"status": "deleted", "id": req.id}


@app.post("/storeVocabulary")
def store_vocabulary(req: VocabularyRequest):
    return {"status": "ok", "count": len(req.words), "words": req.words}


@app.get("/health")
def health_check():
    return {"status": "ok"}
