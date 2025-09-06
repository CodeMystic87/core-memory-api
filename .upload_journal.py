print("✅ upload_journal.py has started running...")

import os
import json
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "dayonememories"

# Always delete the index first (clean slate)
if index_name in [i["name"] for i in pc.list_indexes()]:
    print(f"🗑 Deleting existing index '{index_name}'...")
    pc.delete_index(index_name)

# Recreate index
print(f"📌 Creating fresh index '{index_name}'...")
pc.create_index(
    name=index_name,
    dimension=1536,  # text-embedding-3-small/large
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region=os.getenv("PINECONE_ENV")  # should be "us-east-1"
    )
)

# Connect to new index
index = pc.Index(index_name)

# Path to your JSONL file
file_path = "journal_with_tags_and_categories.jsonl"

print("📂 Checking for journal file...")
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

# Load journal entries
with open(file_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"📖 Loaded {len(entries)} journal entries")

# Process in batches
batch_size = 50
for i in range(0, len(entries), batch_size):
    batch = entries[i:i+batch_size]

    # Extract texts
    texts = [entry["text"] for entry in batch]

    # Generate embeddings
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    # Prepare Pinecone vectors
    vectors = []
    for entry, emb in zip(batch, embeddings.data):
        vectors.append({
            "id": entry["id"],
            "values": emb.embedding,
            "metadata": {
                "tag": entry.get("tag", ""),
                "category": entry.get("category", ""),
                "text": entry["text"]
            }
        })

    # Upload to Pinecone
    index.upsert(vectors=vectors)
    print(f"⬆️ Uploaded {len(vectors)} entries")

print("🎉 Upload complete!")
print("✅ All journal entries uploaded successfully.")
import sys
sys.exit(0)

