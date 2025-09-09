import os
import json
from datetime import datetime
from openai import OpenAI
import pinecone

print("🚀 Running reset_and_migrate.py...")

# === Initialize OpenAI + Pinecone ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

# === 1. Wipe the index ===
print("🧹 Deleting ALL entries from core-memory...")
index.delete(delete_all=True)
print("✅ Index cleared!")

# === Helper: create embeddings ===
def embed_text(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# === Helper: normalize date ===
def normalize_date(date_str):
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d"), dt.strftime("%B %d, %Y")
    except:
        return None, date_str  # fallback

# === 2. Upload entries from a file ===
def upload_entries(path, prefix):
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return 0

    with open(path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f]

    print(f"📦 Uploading {len(entries)} entries from {path}...")

    vectors = []
    for i, entry in enumerate(entries):
        text = entry.get("text", "")
        date_str, date_friendly = normalize_date(entry.get("date", ""))

        emb = embed_text(text)

        vectors.append({
            "id": entry.get("id", f"{prefix}-{i}"),
            "values": emb,
            "metadata": {
                "title": entry.get("title", f"{prefix.title()} Entry {i}"),
                "text": text,
                "tags": entry.get("tags", []),
                "categories": entry.get("categories", []),
                "date": date_str,
                "date_friendly": date_friendly,
                "source": prefix
            }
        })

        # Batch upload every 50
        if len(vectors) >= 50:
            index.upsert(vectors=vectors)
            vectors = []

    if vectors:
        index.upsert(vectors=vectors)

    print(f"✅ Uploaded {len(entries)} from {path}")
    return len(entries)

# === 3. Upload Journals ===
count_journals = upload_entries("journal_with_tags_and_categories.jsonl", "journal")

# === 4. Upload Live Memories if available ===
count_live = upload_entries("live_memories.jsonl", "live")

print(f"🎉 Migration complete! Uploaded {count_journals} journals and {count_live} live memories.")
