from fastapi import FastAPI, Request
from pydantic import BaseModel
import os
from pinecone import Pinecone
from openai import OpenAI

# Initialize FastAPI
app = FastAPI(title="Core Memory API")

# Load environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to Pinecone index (make sure index is already created in console)
INDEX_NAME = "core-memory"
index = pc.Index(INDEX_NAME)

# Request model
class MemoryRequest(BaseModel):
    text: str
    tags: list[str] = []

# Store memory endpoint
@app.post("/store_memory")
def store_memory(request: MemoryRequest):
    embedding = client.embeddings.create(
        input=request.text,
        model="text-embedding-3-small"
    ).data[0].embedding

    index.upsert([
        ("mem_" + str(hash(request.text)), embedding, {"text": request.text, "tags": request.tags})
    ])

    return {"status": "✅ Memory stored", "text": request.text, "tags": request.tags}

# Query memory endpoint
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

@app.post("/query_memory")
def query_memory(request: QueryRequest):
    embedding = client.embeddings.create(
        input=request.query,
        model="text-embedding-3-small"
    ).data[0].embedding

    results = index.query(
        vector=embedding,
        top_k=request.top_k,
        include_metadata=True
    )

    matches = [
        {"text": m["metadata"]["text"], "score": m["score"], "tags": m["metadata"].get("tags", [])}
        for m in results["matches"]
    ]

    return {"results": matches}

