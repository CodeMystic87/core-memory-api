import os
import json
from openai import OpenAI
import pinecone
from datetime import datetime

print("🚀 upload_journal.py has started...")

# === Initialize OpenAI ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Initialize Pinecone ===
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

# === Path to your journal file (already in GitHub repo) ===
file_path = "journal_with_tags_and_categories.jsonl"

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

# === Load entries ===
with open(file_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"📖 Loaded {len(entries)} journal entries.")

# === Normalize date ===
def normalize_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d"), dt.strftime("%B %d, %Y")
    except Exception:
        return None, date_str  # fallback if invalid

# === Upload in batches ===
batch_size = 50
uploaded = 0
batch = []

for i, entry in enumerate(entries):
    text = entry.get("text", "").strip()
    if not text:
        continue

    title = entry.get("title", f"Journal Entry {i}")
    tags = entry.get("tags", [])
    categories = entry.get("categories", [])
    raw_date = entry.get("date", "")

    date_iso, date_friendly = normalize_date(raw_date)

    # Create embedding
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=f"{title}\n{text}"
    ).data[0].embedding

    # Build metadata
    metadata = {
        "type": "journal",
        "title": title,
        "text": text,
        "tags": tags,
        "categories": categories,
        "date": date_iso,
        "date_friendly": date_friendly,
    }

    # Use stable ID (date + index)
    vector_id = f"journal-{date_iso or 'unknown'}-{i}"

    batch.append({
        "id": vector_id,
        "values": embedding,
        "metadata": metadata
    })

    # Upload every 50
    if len(batch) >= batch_size:
        index.upsert(vectors=batch)
        uploaded += len(batch)
        print(f"✅ Uploaded {uploaded}/{len(entries)} so far...")
        batch = []

# Upload remaining
if batch:
    index.upsert(vectors=batch)
    uploaded += len(batch)

print(f"🎉 Finished! Uploaded {uploaded}/{len(entries)} journal entries.")
