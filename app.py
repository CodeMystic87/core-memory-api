from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import os
import uuid

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI + Pinecone
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Pinecone index name (make sure it matches your dashboard)
INDEX_NAME = os.getenv("INDEX_NAME", "core-memory")

# Create Pinecone index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # embedding size for text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# Request models
class StoreMemoryRequest(BaseModel):
    text: str
    metadata: Optional[dict] = None

class SearchMemoryRequest(BaseModel):
    query: str
    topk: int = 5

# Store Memory endpoint
@app.post("/storeMemory")
def store_memory(req: StoreMemoryRequest):
    # Create embedding
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.text
    ).data[0].embedding

    # Unique ID for this memory
    memory_id = str(uuid.uuid4())

    # Store in Pinecone
    index.upsert([
        {
            "id": memory_id,
            "values": embedding,
            "metadata": {"text": req.text, **(req.metadata or {})}
        }
    ])

    return {"message": "Memory stored", "id": memory_id}

# Search Memories endpoint
@app.post("/searchMemories")
def search_memories(req: SearchMemoryRequest):
    # Embed query
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.query
    ).data[0].embedding

    # Query Pinecone
    results = index.query(
        vector=embedding,
        top_k=req.topk,
        include_metadata=True
    )

    matches = [
        {
            "text": match["metadata"]["text"],
            "score": match["score"]
        }
        for match in results["matches"]
    ]

    return {"results": matches}
