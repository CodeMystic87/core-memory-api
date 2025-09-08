# ---------- Store / Update / Delete Endpoints ----------

class StoreMemoryRequest(BaseModel):
    text: str
    tags: Optional[List[str]] = []
    kind: Optional[str] = "note"
    title: Optional[str] = None
    mood: Optional[str] = None
    people: Optional[List[str]] = []
    activities: Optional[List[str]] = []
    keywords: Optional[List[str]] = []
    meta: Optional[dict] = {}

@app.post("/storeMemory")
async def store_memory(req: StoreMemoryRequest):
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.text
    ).data[0].embedding

    mem_id = hashlib.sha256(req.text.encode()).hexdigest()
    metadata = req.dict()

    index.upsert([(mem_id, embedding, metadata)])
    return {"status": "stored", "memory": mem_id, "tags": req.tags}


class UpdateMemoryRequest(BaseModel):
    id: str
    text: Optional[str] = None
    tags: Optional[List[str]] = []
    mood: Optional[str] = None
    people: Optional[List[str]] = []
    activities: Optional[List[str]] = []
    keywords: Optional[List[str]] = []
    meta: Optional[dict] = {}

@app.post("/updateMemory")
async def update_memory(req: UpdateMemoryRequest):
    if not req.text:
        return {"status": "error", "message": "Text is required for update"}

    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=req.text
    ).data[0].embedding

    metadata = req.dict()
    index.upsert([(req.id, embedding, metadata)])
    return {"status": "updated", "memory": req.id, "tags": req.tags}


class DeleteMemoryRequest(BaseModel):
    id: str

@app.post("/deleteMemory")
async def delete_memory(req: DeleteMemoryRequest):
    index.delete(ids=[req.id])
    return {"status": "deleted", "deleted_id": req.id}
