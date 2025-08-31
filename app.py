from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import httpx
from openai import OpenAI
from pinecone import Pinecone

# ---------- Init ----------
app = FastAPI(title="Core Memory API")

OPENAI_API_KEY   = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
INDEX_NAME       = "core-memory"          # Pinecone index must be 1536-dim
EMBED_MODEL      = "text-embedding-3-small"

# Force a clean HTTP client (no proxies)
http_client = httpx.Client(proxies=None, timeout=30)

client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ---------- Health ----------
@app.get("/")
def root():
    return {"status": "Core Memory API is running!"}

# ---------- Store ----------
@app.post("/store_memory")
async def store_memory(request: Request):
    data = await request.json()
    text = (data.get("text") or "").strip()
    tags = data.get("tags") or []

    if not text:
        return JSONResponse({"error": "Missing 'text'."}, status_code=400)

    emb = client.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

    from datetime import datetime
    mem_id = f"mem_{datetime.utcnow().isoformat()}"

    index.upsert([(mem_id, emb, {"text": text, "tags": tags})])

    return {"ok": True, "id": mem_id, "tags": tags}

# ---------- Query ----------
@app.post("/query_memory")
async def query_memory(request: Request):
    data = await request.json()
    query = (data.get("query") or "").strip()
    top_k = int(data.get("top_k") or 5)
    topic = data.get("topic")  # optional tag filter

    if not query:
        return JSONResponse({"error": "Missing 'query'."}, status_code=400)

    qemb = client.embeddings.create(model=EMBED_MODEL, input=query).data[0].embedding

    pinecone_filter = None
    if topic:
        topics = [topic] if isinstance(topic, str) else topic
        pinecone_filter = {"tags": {"$in": topics}}

    results = index.query(
        vector=qemb,
        top_k=top_k,
        include_metadata=True,
        filter=pinecone_filter
    )

    matches = []
    for m in results.get("matches", []):
        matches.append({
            "id": m.get("id"),
            "score": m.get("score"),
            "text": m.get("metadata", {}).get("text"),
            "tags": m.get("metadata", {}).get("tags", [])
        })

    return {"matches": matches}
