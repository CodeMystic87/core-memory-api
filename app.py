from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone
import os

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "core-memory")

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Initialize FastAPI
app = FastAPI()

# Request model for /searchMemories
class SearchRequest(BaseModel):
    query: str
    topk: int = 5

# Response model
class MemorySnippet(BaseModel):
    text: str
    score: float

@app.post("/searchMemories")
def search_memories(req: SearchRequest):
    try:
        # Step 1: Embed query with OpenAI
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

        # Step 3: Format output
        memories = [
            {
                "text": match["metadata"].get("text", ""),
                "score": match["score"]
            }
            for match in results["matches"]
        ]

        return {"results": memories}

    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health_check():
    return {"status": "ok"}
