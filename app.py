import os
import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI

# Initialize FastAPI
app = FastAPI()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "core-memory")

# Initialize clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)


# ================================
# Request/Response Models
# ================================
class SearchRequest(BaseModel):
    query: str
    topk: int = 5


class StoreRequest(BaseModel):
    text: str


# ================================
# Search Memories
# ================================
@app.post("/searchMemories")
def search_memories(req: SearchRequest):
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.query
    ).data[0].embedding

    results = index.query(
        vector=embedding,
        top_k=req.topk,
        include_metadata=True
    )

    memories = [
        {"text": match["metadata"]["text"], "score": match["score"]}
        for match in results["matches"]
    ]

    return {"results": memories}


# ================================
# Store Memory (NEW)
# ================================
@app.post("/storeMemory")
def store_memory(req: StoreRequest):
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.text
    ).data[0].embedding

    # Generate unique ID
    mem_id = str(uuid.uuid4())

    # Insert into Pinecone
    index.upsert([
        {
            "id": mem_id,
            "values": embedding,
            "metadata": {"text": req.text}
        }
    ])

    return {"status": "ok", "id": mem_id, "text": req.text}
