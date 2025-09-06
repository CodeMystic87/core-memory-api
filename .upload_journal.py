import os
import json
import pinecone
from openai import OpenAI

print("✅ upload_journal.py has started running...")

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone (v3 does NOT use environment anymore)
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to your Pinecone index using the host URL from the dashboard
index = pinecone.Index(
    host="https://dayonememories-02h73am.svc.aped-4627-b74a.pinecone.io"
)

# Path to your JSONL file
file_path = "journal_with_tags_and_categories.jsonl"

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

# Read entries from JSONL
with open(file_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

# Batch upload
batch_size = 32
for i in range(0, len(entries), batch_size):
    batch = entries[i:i + batch_size]

    texts = [entry["text"] for entry in batch]
    embeddings = client.embeddings.create(
        input=texts,
        model="text-embedding-ada-002"  # or "text-embedding-3-large" for higher quality
    )

    vectors = []
    for j, entry in enumerate(batch):
        vectors.append({
            "id": f"entry-{i+j}",
            "values": embeddings.data[j].embedding,
            "metadata": {
                "tag": entry.get("tag"),
                "category": entry.get("category"),
                "text": entry.get("text")
            }
        })

    index.upsert(vectors=vectors)

print("🎉 Journal upload completed successfully!")
