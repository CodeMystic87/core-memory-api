from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import os
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

# ---------- Endpoints ----------
from typing import Optional, List
from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: Optional[str] = None              # ✅ Now optional, defaults to None
    topk: int = 10
    date: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    kinds: Optional[List[str]] = []
    tags: Optional[List[str]] = []
    tags_contains_any: Optional[List[str]] = []
    tags_contains_all: Optional[List[str]] = []
    people_contains_any: Optional[List[str]] = []
    mood_contains_any: Optional[List[str]] = []
    activities_contains_any: Optional[List[str]] = []
    include_text: bool = True
    sort_by: str = "newest"
    summarize: bool = False





@app.get("/health")
async def health_check():
    return {"status": "ok"}
