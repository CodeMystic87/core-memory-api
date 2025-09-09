import os
import json
from openai import OpenAI
from pinecone import Pinecone

print("🚀 Running migrate_clean_journal.py...")

# Initialize OpenAI + Pinecone
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

# Path to your journal file
file_path = "journal_with_tags_and_categories.jsonl"

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

# 🔄 Delete only pre-September journal entries
print("⚠️ Clearing journal entries before 2025-09-01...")
index.delete(filter={"date": {"$lt": "2025-09-01"}})
print("✅ Old journal entries cleared. Live memories are safe.")

# Load entries
entries = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            entry = json.loads(line)
            entries.append(entry)
        except Exception as e:
            print(f"⚠️ Skipping invalid line: {e}")

print(f"✅ Loaded {len(entries)} journal entries")

# Upload in batches
batch_size = 100
uploaded_count = 0

for i in range(0, len(entries), batch_size):
    batch = entries[i:i+batch_size]
    vectors = []

    for entry in batch:
        try:
            text = str(entry.get("text", "")).strip()
            if not text:
                continue

            embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding

            metadata = {
                "text": text,
                "title": entry.get("title", "Untitled"),
                "date": entry.get("date", "1970-01-01"),  # fallback
                "tags": entry.get("tags", []),
                "category": entry.get("category", "journal")
            }

            vectors.append({
                "id": str(entry.get("id", f"entry-{i}")),
                "values": embedding,
                "metadata": metadata
            })

        except Exception as e:
            print(f"❌ Skipping entry due to error: {e}")

    if vectors:
        index.upsert(vectors)
        uploaded_count += len(vectors)
        print(f"✅ Uploaded {len(vectors)} entries in batch {i//batch_size+1}")

print(f"\n🎉 Finished migration! Total uploaded: {uploaded_count}/{len(entries)}")
