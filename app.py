from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
from openai import OpenAI
import pinecone

# Initialize FastAPI
app = FastAPI(title="Core Memory API")

# Load environment variables
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
APP_TOKEN = os.environ["APP_TOKEN"]
INDEX_NAME = "core-memory"

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# --- Auth helper ---
def verify_token(request: Request):
    token = request.headers.get("Authorization")
    if token != f"Bearer {APP_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return True

# --- Routes ---
@app.get("/")
def root():
    return {"status": "Core Memory API is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/status")
def status():
    try:
        stats = index.describe_index_stats()
        return {"status": "ok", "pinecone_index": INDEX_NAME, "stats": stats}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.post("/store_memory")
async def store_memory(request: Request, authorized: bool = Depends(verify_token)):
    data = await request.json()
    text = data.get("text", "")

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

    index.upsert(vectors=[("mem1", embedding, {"text": text})])

    return {"status": "memory stored", "text": text}

@app.post("/query_memory")
async def query_memory(request: Request, authorized: bool = Depends(verify_token)):
    data = await request.json()
    query = data.get("query", "")

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    return {"matches": results["matches"]}
