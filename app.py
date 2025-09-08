from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
import hashlib
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
class MemoryRequest(BaseModel):
    text: str
    tags: Optional[List[str]] = []

class VocabularyRequest(BaseModel):
    words: List[str]

class SearchRequest(BaseModel):
    query: str
    topk: int = 5
    tags: Optional[List[str]] = []

# ---------- Endpoints ----------
@app.post("/searchMemories")
async def search_memories(req: SearchRequest):
    embedding = None

    # Case 1: Query exists → generate embedding from OpenAI
    if req.query:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=req.query
        ).data[0].embedding
    else:
        # Case 2: No query → skip embeddings, use dummy vector
        # "text-embedding-3-small" has 1536 dimensions
        embedding = [0.0] * 1536

    # Build filters
    pinecone_filter = None
    if req.tags:
        pinecone_filter = {"tags": {"$in": req.tags}}

    # Run Pinecone query
    results = index.query(
        vector=embedding,
        top_k=req.topk,
        include_metadata=True,
        filter=pinecone_filter
    )

    matches = [
        {
            "id": match["id"],
            "score": match["score"],
            "metadata": match["metadata"]
        }
        for match in results["matches"]
    ]

    return {"results": matches}




@app.get("/health")
async def health_check():
    return {"status": "ok"}
