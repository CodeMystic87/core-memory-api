import os
import json
import pinecone
from openai import OpenAI

print("✅ upload_journal.py has started running...")

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to your *existing* index (from Pinecone dashboard)
index = pc.Index(
    host="https://dayonememories-02h73am.svc.aped-4627-b74a.pinecone.io"
)

# Path to your JSONL file
file_path = "journal_with_tags_and_categories.jsonl"

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

# Load journal entries
with open(file_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"📓 Loaded {len(entries)} journal entries")

# Batch size for uploading
batch_size = 32

for i in range(0, len(entries), batch_size):
    batch = entries[i:i + batch_size]

    # Generate embeddings
    texts = [entry["text"] for entry in batch]
    embeddings = client.embeddings.create(
        input=texts,
        model="text-embedding-ada-002"   # can upgrade to "text-embedding-3-large" later
    )

    # Prepare vectors
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

    # Upload to Pinecone
    index.upsert(vectors=vectors)
    print(f"✅ Uploaded batch {i // batch_size + 1} ({len(batch)} entries)")

print("🎉 Journal upload completed successfully!")
