

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import fastapi
import pydantic
import uvicorn
import openai
import httpx
import json
import os
import uuid

try:
    import pinecone
    pinecone_version = pinecone.__version__
except ImportError:
    pinecone_version = "not installed"

app = FastAPI(title="CoreMemory API", version="1.0.0")

# --- Local Fallback Cache (JSON so it survives restarts) ---
CACHE_FILE = "memory_cache.json"

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r") as f:
        memory_cache = json.load(f)
else:
    memory_cache = {"memories": [], "vocab": []}


def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(memory_cache, f, indent=2)


# --- Models ---
class Meta(BaseModel):
    datetime_iso: str
    timezone: str
    version: str


class MemoryRequest(BaseModel):
    id: Optional[str] = None
    text: str
    tags: Optional[List[str]] = []
    kind: str  # journal, note, task, idea, milestone
    mood: Optional[str] = None
    people: Optional[List[str]] = []
    activities: Optional[List[str]] = []
    keywords: Optional[List[str]] = []
    meta: Meta


class SearchRequest(BaseModel):
    query: Optional[str] = None
    date: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    kinds: Optional[List[str]] = []
    tags_contains: Optional[List[str]] = []
    tags_contains_any: Optional[List[str]] = []
    tags_contains_all: Optional[List[str]] = []
    people_contains_any: Optional[List[str]] = []
    mood_contains_any: Optional[List[str]] = []
    activities_contains_any: Optional[List[str]] = []
    include_text: Optional[bool] = True
    sort_by: Optional[str] = "newest"  # relevance, newest, oldest
    summarize: Optional[bool] = False


class UpdateRequest(BaseModel):
    id: str
    text: Optional[str] = None
    tags: Optional[List[str]] = None
    mood: Optional[str] = None
    people: Optional[List[str]] = None
    activities: Optional[List[str]] = None
    keywords: Optional[List[str]] = None


class DeleteRequest(BaseModel):
    id: str


class VocabularyRequest(BaseModel):
    words: List[str]


# --- Endpoints ---
@app.post("/storeMemory")
def store_memory(req: MemoryRequest):
    entry_id = req.id or str(uuid.uuid4())
    entry = req.dict()
    entry["id"] = entry_id

    memory_cache["memories"].append(entry)
    save_cache()

    return {"status": "ok", "id": entry_id, "stored": entry}


@app.post("/searchMemories")
def search_memories(req: SearchRequest):
    results = []

    for mem in memory_cache["memories"]:
        if req.query and req.query.lower() not in mem["text"].lower():
            continue
        if req.kinds and mem["kind"] not in req.kinds:
            continue
        results.append(mem)

    # Sort
    if req.sort_by == "newest":
        results.sort(key=lambda x: x["meta"]["datetime_iso"], reverse=True)
    elif req.sort_by == "oldest":
        results.sort(key=lambda x: x["meta"]["datetime_iso"])

    return {"results": results}


@app.post("/updateMemory")
def update_memory(req: UpdateRequest):
    for mem in memory_cache["memories"]:
        if mem["id"] == req.id:
            if req.text is not None:
                mem["text"] = req.text
            if req.tags is not None:
                mem["tags"] = req.tags
            if req.mood is not None:
                mem["mood"] = req.mood
            if req.people is not None:
                mem["people"] = req.people
            if req.activities is not None:
                mem["activities"] = req.activities
            if req.keywords is not None:
                mem["keywords"] = req.keywords
            save_cache()
            return {"status": "ok", "updated": mem}
    return {"status": "error", "message": "Memory not found"}


@app.post("/deleteMemory")
def delete_memory(req: DeleteRequest):
    for mem in memory_cache["memories"]:
        if mem["id"] == req.id:
            memory_cache["memories"].remove(mem)
            save_cache()
            return {"status": "ok", "deleted_id": req.id}
    return {"status": "error", "message": "Memory not found"}


@app.post("/storeVocabulary")
def store_vocabulary(req: VocabularyRequest):
    memory_cache["vocab"].extend(req.words)
    save_cache()
    return {"status": "ok", "vocab_size": len(memory_cache["vocab"])}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/testNoConfirm")
def test_no_confirm():
    return {"message": "Test endpoint is working"}


# --- Version Debug Endpoint ---
@app.get("/version")
def version_check():
    return {
        "fastapi": fastapi.__version__,
        "pydantic": pydantic.__version__,
        "uvicorn": uvicorn.__version__,
        "openai": getattr(openai, "__version__", "unknown"),
        "httpx": httpx.__version__,
        "pinecone-client": pinecone_version
    }


