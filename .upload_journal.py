import os
import json
from openai import OpenAI
from pinecone import Pinecone

print("✅ upload_journal.py has started running...")

# --- Initialize OpenAI ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Initialize Pinecone ---
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "dayonememories"

if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": os.getenv("PINECONE_ENV", "us-east-1")
            }
        }
    )

index = pc.Index(index_name)

# --- Load journal entries ---
file_path = "journal_with_tags_and_categories.jsonl"
print(f"📂 Checking for journal file: {file_path}")

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

with open(file_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"✅ Loaded {len(entries)} journal entries")

# --- Process entries in batches ---
batch_size = 100
for i in range(0, len(entries), batch_size):
    batch = entries[i:i + batch_size]
    print(f"📦 Processing batch {i // batch_size + 1} with {len(batch)} entries...")

    texts = [entry["text"] for entry in batch]

    # ✅ Ensure correct API call for embeddings
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    vectors = [
        (
            f"entry-{i+j}",
            embeddings.data[j].embedding,
            {
                "tag": entry.get("tag", ""),
                "category": entry.get("category", ""),
                "text": entry["text"]
            }
        )
        for j, entry in enumerate(batch)
    ]

    index.upsert(vectors=vectors)

print("🎉 Finished uploading all journal entries to Pinecone!")
