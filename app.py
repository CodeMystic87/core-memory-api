# --- FastAPI app ---
app = FastAPI()

# Request model
class SearchRequest(BaseModel):
    query: str
    topk: int = 5

# Response model
class MemoryResult(BaseModel):
    text: str
    score: float

class SearchResponse(BaseModel):
    results: list[MemoryResult]

@app.post("/searchMemories", response_model=SearchResponse)
def search_memories(req: SearchRequest):
    # Step 1: Embed query
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=req.query
    ).data[0].embedding

    # Step 2: Query Pinecone
    results = index.query(
        vector=embedding,
        top_k=req.topk,
        include_metadata=True
    )

    # Step 3: Format response
    memories = [
        {"text": match["metadata"]["text"], "score": match["score"]}
        for match in results["matches"]
    ]

    return {"results": memories}
