print("✅ upload_journal.py has started running...")

import os
import json
import pinecone
from openai import OpenAI

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")   # should be "us-east-1" for your index
)

index_name = "dayonememories"

# Auto-create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    print(f"📌 Creating index '{index_name}'...")
    pinecone.create_index(
        name=index_name,
        dimension=1536,        # embedding size for text-embedding-3-small/large
        metric="cosine",
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_ENV")  # us-east-1
        )
    )
else:
    print(f"✅ Index '{index_name}' already exists")

# Connect to index
index = pinecone.Index(index_name)

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

# Process entries in batches
batch_size = 50
for i in range(0, len(entries), batch_size):
    batch = entries[i:i+batch_size]

    # Extract texts
    texts = [entry["text"] for entry in batch]

    # Create embeddings
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",   # or "text-embedding-3-large"
        input=texts
    )

    # Format vectors for Pinecone
    vectors = []
    for entry, emb in zip(batch, embeddings.data):
        vectors.append({
            "id": entry["id"],  # unique ID for the entry
            "values": emb.embedding,
            "metadata": {
                "tag": entry.get("tag", ""),
                "category": entry.get("category", ""),
                "text": entry["text"]
            }
        })

    # Upload to Pinecone
    index.upsert(vectors)
    print(f"⬆️ Uploaded {len(vectors)} entries")

print("✅ Upload complete! 🎉")
