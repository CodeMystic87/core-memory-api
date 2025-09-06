import os
import json
from openai import OpenAI
from pinecone import Pinecone

print("🚀 upload_journal.py has started running...")

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("dayonememories")

# Path to your JSONL file
file_path = "journal_with_tags_and_categories.jsonl"

print("📂 Checking for journal file...")
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

# Load journal entries
with open(file_path, "r") as f:
    entries = [json.loads(line) for line in f]

print(f"✅ Loaded {len(entries)} journal entries")

# Batch size for embedding requests
batch_size = 100

for i in range(0, len(entries), batch_size):
    batch = entries[i:i+batch_size]
    texts = [entry["text"] for entry in batch]

    # Generate embeddings
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    embeddings = [item.embedding for item in response.data]

    # Prepare vectors for Pinecone
    vectors = []
    for entry, embedding in zip(batch, embeddings):
        vectors.append({
            "id": entry.get("id", str(hash(entry["text"]))),  # ensure unique IDs
            "values": embedding,
            "metadata": {
                "date": entry.get("date"),
                "tags": entry.get("tags", []),
                "category": entry.get("category", "uncategorized"),
                "text": entry["text"]
            }
        })

    # Upsert into Pinecone
    index.upsert(vectors)
    print(f"✅ Processed batch {i//batch_size + 1} with {len(batch)} entries")

print("🎉 Finished uploading all journal entries to Pinecone!")
