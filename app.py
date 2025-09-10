from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pytz
import uuid

app = FastAPI()

# ----------------------------
# Fallback cache (in-memory)
# ----------------------------
fallback_cache = {}

# ----------------------------
# Request Models
# ----------------------------
class MetaData(BaseModel):
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
    meta: MetaData

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
    include_text: Optional[bool] = False
    sort_by: Optional[str] = "relevance"  # relevance, newest, oldest
    summarize: Optional[bool] = False

class UpdateRequest(BaseModel):
    id: str
    text: Optional[str] = None
    tags: Optional[List[str]] = []
    mood: Optional[str] = None
    people: Optional[List[str]] = []
    activities: Optional[List[str]] = []
    keywords: Optional[List[str]] = []

class DeleteRequest(BaseModel):
    id: str

class VocabularyRequest(BaseModel):
    words: List[str]

# ----------------------------
# Endpoints
# ----------------------------

@app.post("/storeMemory")
def store_memory(req: MemoryRequest):
    memory_id = req.id or str(uuid.uuid4())
    fallback_cache[memory_id] = req.dict()
    return {"status": "stored", "id": memory_id, "memory": req.dict()}

@app.post("/searchMemories")
def search_memories(req: SearchRequest):
    results = []
    for mem_id, mem in fallback_cache.items():
        if req.query and req.query.lower() not in mem["text"].lower():
            continue
        results.append(mem)
    return {"results": results}

@app.post("/updateMemory")
def update_memory(req: UpdateRequest):
    if req.id not in fallback_cache:
        raise HTTPException(status_code=404, detail="Memory not found")
    fallback_cache[req.id].update({k: v for k, v in req.dict().items() if v is not None})
    return {"status": "updated", "id": req.id, "memory": fallback_cache[req.id]}

@app.post("/deleteMemory")
def delete_memory(req: DeleteRequest):
    if req.id not in fallback_cache:
        raise HTTPException(status_code=404, detail="Memory not found")
    deleted = fallback_cache.pop(req.id)
    return {"status": "deleted", "id": req.id, "memory": deleted}

@app.post("/storeVocabulary")
def store_vocabulary(req: VocabularyRequest):
    return {"status": "stored", "words": req.words}

@app.get("/health")
def health_check():
    return {"status": "ok", "time": datetime.now().isoformat()}

@app.get("/testNoConfirm")
def test_no_confirm():
    return {"message": "Test endpoint response"}
