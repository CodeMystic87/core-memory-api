from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

# --- Keys and setup ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_HOST = os.environ.get("PINECONE_HOST", "https://core-memory-02h73am.svc.aped-4627-b74a.pinecone.io")
INDEX_NAME = os.environ.get("INDEX_NAME", "core-memory")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# --- FastAPI app ---
app = FastAPI()

# Request model
class SearchRequest(BaseModel):
    query: str
    topk: int = 5

# Response model
class MemoryResult(BaseModel):
    text: str
    score: float

class SearchResponse(BaseModel):
    results: list[MemoryResult]

@app.post("/searchMemories", response_model=SearchResponse)
def search_memories(req: SearchRequest):
    # Step 1: Embed query
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=req.query
    ).data[0].embedding

    # Step 2: Query Pinecone
    results = index.query(
        vector=embedding,
        top_k=req.topk,
        include_metadata=True
    )

    # Step 3: Format response
    memories = [
        {"text": match["metadata"]["text"], "score": match["score"]}
        for match in results["matches"]
    ]

    return {"results": memories}
