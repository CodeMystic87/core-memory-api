from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os, datetime
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
    tags: Optional[List[str]] = []  # ✅ now supports tags

class VocabularyRequest(BaseModel):
    words: List[str]

class SearchRequest(BaseModel):
    query: str
    topk: int = 5

# ---------- Endpoints ----------
@app.post("/storeMemory")
async def store_memory(req: MemoryRequest):
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.text
    ).data[0].embedding

    # ✅ Attach metadata: text, tags, timestamp
    metadata = {
        "text": req.text,
        "tags": req.tags or [],
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

    # Use unique ID for Pinecone (avoid overwriting)
    item_id = f"mem-{datetime.datetime.utcnow().timestamp()}"

    index.upsert([
        (item_id, embedding, metadata)
    ])
    return {"status": "stored", "memory": req.text, "tags": req.tags}

@app.post("/storeVocabulary")
async def store_vocabulary(req: VocabularyRequest):
    for word in req.words:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=word
        ).data[0].embedding
        metadata = {
            "text": word,
            "tags": ["vocabulary"],
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        index.upsert([
            (f"vocab-{word}-{datetime.datetime.utcnow().timestamp()}", embedding, metadata)
        ])
    return {"status": "stored", "count": len(req.words)}

@app.post("/searchMemories")
async def search_memories(req: SearchRequest):
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.query
    ).data[0].embedding

    results = index.query(
        vector=embedding,
        top_k=req.topk,
        include_metadata=True
    )

    matches = [
        {
            "text": match["metadata"].get("text", ""),
            "tags": match["metadata"].get("tags", []),
            "timestamp": match["metadata"].get("timestamp", ""),
            "score": match["score"]
        }
        for match in results["matches"]
    ]
    return {"results": matches}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
