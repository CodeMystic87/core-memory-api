import os
import json
import hashlib
import datetime
from openai import OpenAI
import pinecone

print("🚀 migrate_journal.py has started...")

# Initialize OpenAI + Pinecone
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

# Path to your JSONL journal
file_path = "journal_with_tags_and_categories.jsonl"

# Load entries
with open(file_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"📒 Loaded {len(entries)} journal entries.")

# Step 1: Delete only journal entries before Sept 1, 2025
print("🧹 Cleaning old journal entries...")
to_delete = []
res = index.query(vector=[0]*1536, top_k=20000, include_metadata=True)
for match in res["matches"]:
    meta = match.get("metadata", {})
    if meta.get("kind") == "journal" and meta.get("date", "9999-12-31") < "2025-09-01":
        to_delete.append(match["id"])

if to_delete:
    print(f"🗑️ Deleting {len(to_delete)} old journal entries...")
    index.delete(ids=to_delete)
else:
    print("✅ No old journal entries to delete.")

# Step 2: Re-upload journals with correct schema
print("⬆️ Uploading journal entries...")

for i, entry in enumerate(entries):
    text = entry.get("text", "").strip()
    title = entry.get("title", "Untitled")
    date_raw = entry.get("date")  # e.g. "2025-08-24"
    tags = entry.get("tags", [])
    categories = entry.get("categories", [])

    # Parse date
    try:
        dt = datetime.datetime.strptime(date_raw, "%Y-%m-%d")
        date_str = dt.strftime("%Y-%m-%d")
        date_friendly = dt.strftime("%B %d, %Y")
    except Exception as e:
        print(f"⚠️ Skipping invalid date {date_raw}: {e}")
        continue

    # Stable ID: journal-YYYYMMDD-hash
    uid = hashlib.md5((date_str + text[:50]).encode("utf-8")).hexdigest()[:8]
    vector_id = f"journal-{date_str}-{uid}"

    # Embedding
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=f"{title}\n{text}"
    ).data[0].embedding

    # Metadata
    metadata = {
        "kind": "journal",
        "title": title,
        "text": text,
        "date": date_str,             # strict format
        "date_friendly": date_friendly,  # human format
        "tags": tags,
        "categories": categories,
    }

    # Upsert
    index.upsert(vectors=[{"id": vector_id, "values": emb, "metadata": metadata}])

    if i % 50 == 0:
        print(f"✅ Uploaded {i+1}/{len(entries)} entries...")

print("🎉 Migration complete! Journals are re-indexed with correct date fields.")
