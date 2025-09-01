from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
import datetime
import dateparser  # NEW: natural language date parsing
import hashlib
from openai import OpenAI
from pinecone import Pinecone

# Initialize FastAPI
app = FastAPI()

# Initialize OpenAI
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("INDEX_NAME", "core-memory")
index = pc.Index(index_name)

# ---------- Request Models ----------
class MemoryRequest(BaseModel):
    text: str
    tags: Optional[List[str]] = []

class VocabularyRequest(BaseModel):
    words: List[str]

class SearchRequest(BaseModel):
    query: str
    topk: int = 5
    tags: Optional[List[str]] = []

# ---------- Helpers ----------
def normalize_dates_from_query(query: str) -> List[str]:
    """
    Look for natural language dates in the query and return standardized YYYY-MM-DD tags.
    """
    possible_tags = []
    today = datetime.date.today()

    # Use dateparser to catch things like "September 1st", "yesterday"
    parsed_date = dateparser.parse(query)
    if parsed_date:
        possible_tags.append(parsed_date.date().isoformat())

    # Handle some common keywords explicitly
    if "today" in query.lower():
        possible_tags.append(today.isoformat())
    if "yesterday" in query.lower():
        possible_tags.append((today - datetime.timedelta(days=1)).isoformat())
    if "last week" in query.lower():
        for i in range(7):
            possible_tags.append((today - datetime.timedelta(days=i)).isoformat())

    return list(set(possible_tags))  # Deduplicate

# ---------- Endpoints ----------
@app.post("/storeMemory")
async def store_memory(req: MemoryRequest):
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.text
    ).data[0].embedding

    metadata = {"text": req.text}
    if req.tags:
        metadata["tags"] = req.tags

    memory_id = hashlib.md5(req.text.encode()).hexdigest()
    index.upsert([(memory_id, embedding, metadata)])
    return {"status": "stored", "memory": req.text, "tags": req.tags}

@app.post("/storeVocabulary")
async def store_vocabulary(req: VocabularyRequest):
    for word in req.words:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=word
        ).data[0].embedding
        word_id = f"vocab-{word}"
        index.upsert([(word_id, embedding, {"text": word})])
    return {"status": "stored", "count": len(req.words)}

@app.post("/searchMemories")
async def search_memories(req: SearchRequest):
    # Generate embedding
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.query
    ).data[0].embedding

    # Normalize query → tags
    inferred_tags = normalize_dates_from_query(req.query)
    all_tags = (req.tags or []) + inferred_tags

    # Run Pinecone query
    results = index.query(
        vector=embedding,
        top_k=req.topk,
        include_metadata=True
    )

    # Filter results by tags if any
    matches = []
    for match in results["matches"]:
        memory_tags = match["metadata"].get("tags", [])
        if not all_tags or any(tag in memory_tags for tag in all_tags):
            matches.append({
                "text": match["metadata"].get("text", ""),
                "tags": memory_tags,
                "score": match["score"]
            })

    return {"results": matches}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
