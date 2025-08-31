from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
from openai import OpenAI
import pinecone
import datetime

# Initialize FastAPI
app = FastAPI(title="Core Memory API")

# Load API keys from environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME = "core-memory"

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pinecone.init(api_key=PINECONE_API_KEY)
index = pinecone.Index(INDEX_NAME)


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
        text = data.get("text", "")
        tags = data.get("tags", [])

        if not text:
            return JSONResponse(content={"error": "No text provided"}, status_code=400)

        # Create embedding for memory text
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        # Unique ID for this memory
        memory_id = f"mem_{datetime.datetime.utcnow().isoformat()}"

        # Store in Pinecone
        index.upsert([
            {
                "id": memory_id,
                "values": embedding,
                "metadata": {
                    "text": text,
                    "tags": tags,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
            }
        ])

        return {"message": "Memory stored successfully", "id": memory_id}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/query_memory")
async def query_memory(request: Request):
    try:
        data = await request.json()
        query = data.get("query", "")
        top_k = data.get("top_k", 5)

        if not query:
            return JSONResponse(content={"error": "No query provided"}, status_code=400)

        # Create embedding for the query
        query_embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        # Query Pinecone index
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        matches = results.get("matches", [])

        if not matches:
            return {"message": "No relevant memories found."}

        # Format results
        formatted = [
            {
                "text": m["metadata"].get("text"),
                "tags": m["metadata"].get("tags", []),
                "score": m.get("score"),
                "timestamp": m["metadata"].get("timestamp")
            }
            for m in matches
        ]

        return {"matches": formatted}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
