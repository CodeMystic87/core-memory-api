from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import os
from openai import OpenAI
import pinecone
import time
import uuid

# Initialize FastAPI
app = FastAPI(title="Core Memory API")

# Load API keys from environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "core-memory"

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

# Connect to Pinecone index
index = pc.Index(INDEX_NAME)

# ------------------------------
# Routes
# ------------------------------

@app.get("/")
def root():
    return {"status": "Core Memory API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/store_memory")
async def store_memory(request: Request):
    try:
        data = await request.json()
        text = data.get("text")
        tags = data.get("tags", [])

        if not text:
            raise HTTPException(status_code=400, detail="`text` field is required.")

        # Embedding model (1536 dims matches Pinecone index)
        EMBED_MODEL = "text-embedding-3-small"

        embedding = client.embeddings.create(
            model=EMBED_MODEL,
            input=text
        ).data[0].embedding

        mem_id = f"mem_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        index.upsert(vectors=[{
            "id": mem_id,
            "values": embedding,
            "metadata": {
                "text": text,
                "tags": tags,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }])

        return {"message": "Memory stored successfully", "id": mem_id}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "store_failed", "detail": str(e)}
        )

@app.post("/query_memory")
async def query_memory(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "")
        top_k = int(data.get("top_k", 3))

        if not query or not isinstance(query, str):
            raise HTTPException(status_code=400, detail="`query` (string) is required.")

        EMBED_MODEL = "text-embedding-3-small"  # must match store_memory

        emb = client.embeddings.create(
            model=EMBED_MODEL,
            input=query
        ).data[0].embedding

        results = index.query(
            vector=emb,
            top_k=top_k,
            include_metadata=True
        )

        matches = []
        for m in results.get("matches", []):
            md = m.get("metadata", {}) or {}
            matches.append({
                "id": m.get("id"),
                "score": m.get("score"),
                "text": md.get("text"),
                "timestamp": md.get("timestamp"),
                "tags": md.get("tags", []),
            })

        return {"matches": matches}

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "query_failed", "detail": str(e)}
        )
