import os
import json
from openai import OpenAI
import pinecone

print("🚀 upload_journal.py has started running...")

# === Initialize OpenAI ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Initialize Pinecone ===
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "dayonememories"

# Create index if it doesn’t exist
if index_name not in [idx.name for idx in pc.list_indexes()]:
    print(f"🆕 Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=1536,              # must match embedding size
        metric="cosine",
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_ENV", "us-east-1")
        )
    )

index = pc.Index(index_name)

# === Load Journal File ===
file_path = "journal_with_tags_and_categories.jsonl"

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

entries = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"⚠️ Skipping invalid JSON line: {e}")

print(f"✅ Loaded {len(entries)} journal entries.")

# === Process in Batches ===
batch_size = 100
for i in range(0, len(entries), batch_size):
    batch = entries[i:i+batch_size]
    vectors = []

    for j, entry in enumerate(batch):
        text = entry.get("text", "").strip()
        if not text:
            continue

        # Get embedding
        response = client.embeddings.create(
            model="text-embedding-3-small",  # can switch to -large for more accuracy
            input=text
        )
        vector = response.data[0].embedding

        vectors.append({
            "id": str(entry.get("id", f"entry-{i+j}")),
            "values": vector,
            "metadata": {
                "text": text,
                "tags": entry.get("tags", []),
                "category": entry.get("category", "")
            }
        })

    # Upload to Pinecone
    if vectors:
        index.upsert(vectors=vectors)
        print(f"⬆️ Uploaded {len(vectors)} vectors (batch {i//batch_size + 1})")

print("🎉 Finished uploading all journal entries to Pinecone!")
