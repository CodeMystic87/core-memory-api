import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Initialize FastAPI
app = FastAPI(title="Core Memory API")

# Load env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("INDEX_NAME", "core-memory")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
APP_TOKEN = os.getenv("APP_TOKEN")

# Init clients
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Ensure index exists
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # matches OpenAI embed model
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

index = pc.Index(INDEX_NAME)

# Models
class MemoryInput(BaseModel):
    text: str
    tags: list[str] = []

class QueryInput(BaseModel):
    query: str
    top_k: int = 3

# Middleware for token auth
@app.middleware("http")
async def check_token(request: Request, call_next):
    token = request.headers.get("x-app-token")
    if token != APP_TOKEN:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})
    return await call_next(request)

# Routes
@app.get("/")
def root():
    return {"status": "✅ Core Memory API running!"}

@app.post("/store")
def store_memory(mem: MemoryInput):
    embedding = client.embeddings.create(
        model=EMBED_MODEL,
        input=mem.text
    ).data[0].embedding

    mem_id = f"mem-{os.urandom(6).hex()}"
    index.upsert([(mem_id, embedding, {"text": mem.text, "tags": mem.tags})])
    return {"status": "✅ stored", "id": mem_id, "tags": mem.tags}

@app.post("/query")
def query_memory(query: QueryInput):
    embedding = client.embeddings.create(
        model=EMBED_MODEL,
        input=query.query
    ).data[0].embedding

    results = index.query(
        vector=embedding,
        top_k=query.top_k,
        include_metadata=True
    )

    return {"matches": results["matches"]}
