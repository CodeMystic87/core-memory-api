import os
import json
import openai
import pinecone

# 1. Load API keys (make sure to set these in Render → Environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")  # e.g., "us-west-2"
)

# 2. Connect to your Pinecone index
index = pinecone.Index("dayonememories")

# 3. Load JSONL file (must be in your repo or uploaded to the server)
with open("journal_with_tags_and_categories.jsonl", "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

# 4. Upload entries in batches
batch_size = 50
for i in range(0, len(entries), batch_size):
    batch = entries[i:i+batch_size]

    # Create embeddings with OpenAI
    texts = [entry["text"] for entry in batch]
    embeddings = openai.Embedding.create(
        input=texts,
        model="text-embedding-ada-002"  # or "text-embedding-3-large" if you want higher quality
    )["data"]

    # Format for Pinecone
    vectors = []
    for j, entry in enumerate(batch):
        vectors.append((
            entry["id"],  # unique ID for the entry
            embeddings[j]["embedding"],  # the embedding vector
            {
                "date": entry["date"],
                "tags": entry["tags"],
                "category": entry["category"],
                "text": entry["text"]
            }
        ))

    # Upsert into Pinecone
    index.upsert(vectors)

print("✅ Upload complete. All journal entries inserted into Pinecone.")
