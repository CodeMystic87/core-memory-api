from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
from openai import OpenAI
import pinecone

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

@app.get("/")
def root():
    return {"status": "Core Memory API is running!"}

# ✅ Health check route (for Render)
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/store_memory")
async def store_memory(request: Request):
    data = await request.json()
    text = data.get("text", "")

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

    index.upsert(
        vectors=[("memory1", embedding, {"text": text})]
    )

    return {"message": "Memory stored successfully"}

@app.post("/query_memory")
async def query_memory(request: Request):
    data = await request.json()
    query = data.get("query", "")

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    results = index.query(
        vector=embedding,
        top_k=3,
        include_metadata=True
    )

    return {"matches": results["matches"]}
