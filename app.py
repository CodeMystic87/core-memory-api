rom fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

app = FastAPI()

# --- Setup Clients ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "core-memory")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# --- Request Schema ---
class SearchRequest(BaseModel):
    query: str
    topK: int = 5

# --- Search Endpoint ---
@app.post("/searchMemories")
def search_memories(req: SearchRequest):
    # Step 1: Embed query
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=req.query
    ).data[0].embedding

    # Step 2: Query Pinecone
    results = index.query(
        vector=embedding,
        top_k=req.topK,
        include_metadata=True
    )

    # Step 3: Format response
    memories = [
        {
            "text": match["metadata"]["text"],
            "score": match["score"]
        }
        for match in results["matches"]
    ]

    return {"results": memories}
