from datetime import datetime

# ---------- Request Models for Memory Ops ----------
class StoreMemoryRequest(BaseModel):
    text: str
    tags: Optional[List[str]] = []
    kind: str = "note"
    title: Optional[str] = None
    mood: Optional[str] = None
    people: Optional[List[str]] = []
    activities: Optional[List[str]] = []
    keywords: Optional[List[str]] = []

class UpdateMemoryRequest(BaseModel):
    id: str
    text: Optional[str] = None
    tags: Optional[List[str]] = None
    mood: Optional[str] = None
    people: Optional[List[str]] = None
    activities: Optional[List[str]] = None
    keywords: Optional[List[str]] = None

class DeleteMemoryRequest(BaseModel):
    id: str

# ---------- Endpoints ----------
@app.post("/storeMemory")
async def store_memory(req: StoreMemoryRequest):
    # Generate embedding for the text
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.text
    ).data[0].embedding

    # Use a hash of the text for stable IDs
    memory_id = hashlib.sha256(req.text.encode("utf-8")).hexdigest()

    # Metadata to store in Pinecone
    metadata = {
        "text": req.text,
        "tags": req.tags,
        "kind": req.kind,
        "title": req.title,
        "mood": req.mood,
        "people": req.people,
        "activities": req.activities,
        "keywords": req.keywords,
        "created_at": datetime.utcnow().isoformat()
    }

    # Upsert into Pinecone
    index.upsert([
        {
            "id": memory_id,
            "values": embedding,
            "metadata": metadata
        }
    ])

    return {
        "status": "saved",
        "memory": memory_id,
        "tags": req.tags
    }

@app.post("/updateMemory")
async def update_memory(req: UpdateMemoryRequest):
    # Fetch the old metadata
    old = index.fetch([req.id])
    if not old or req.id not in old["vectors"]:
        return {"status": "not found", "id": req.id}

    old_metadata = old["vectors"][req.id]["metadata"]

    # Merge updates into metadata
    new_metadata = {**old_metadata}
    if req.text:
        new_metadata["text"] = req.text
    if req.tags is not None:
        new_metadata["tags"] = req.tags
    if req.mood is not None:
        new_metadata["mood"] = req.mood
    if req.people is not None:
        new_metadata["people"] = req.people
    if req.activities is not None:
        new_metadata["activities"] = req.activities
    if req.keywords is not None:
        new_metadata["keywords"] = req.keywords

    # If text changed, regenerate embedding
    if req.text:
        embedding = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=req.text
        ).data[0].embedding
    else:
        embedding = old["vectors"][req.id]["values"]

    # Upsert updated record
    index.upsert([
        {
            "id": req.id,
            "values": embedding,
            "metadata": new_metadata
        }
    ])

    return {"status": "updated", "memory": req.id, "tags": new_metadata.get("tags", [])}

@app.post("/deleteMemory")
async def delete_memory(req: DeleteMemoryRequest):
    index.delete(ids=[req.id])
    return {"status": "deleted", "id": req.id}
