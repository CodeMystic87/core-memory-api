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

@app.post("/store_memory")
async def store_memory(request: Request):
    data = await request.json()
    text = data.get("text", "")

    # Get embedding
    embedding = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Store in Pinecone
    index.upsert([("mem_1", embedding, {"text": text})])

    return {"status": "success", "stored_text": text}

@app.post("/query_memory")
async def query_memory(request: Request):
    data = await request.json()
    query = data.get("query", "")

    # Get embedding
    embedding = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    # Query Pinecone
    results = index.query(vector=embedding, top_k=3, include_metadata=True)

    return {"matches": results["matches"]}

