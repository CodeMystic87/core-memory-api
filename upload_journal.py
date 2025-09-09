import os
import sys
import json
from openai import OpenAI
import pinecone

print("🚀 upload_journal.py has started...")

# === Load API keys ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

# === Get input file path ===
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = os.path.join("core_memory_api", "journal_with_tags_and_categories.jsonl")

print(f"📂 Using journal file: {input_file}")

if not os.path.exists(input_file):
    raise FileNotFoundError(f"❌ File not found: {input_file}")

# === Upload entries ===
with open(input_file, "r") as f:
    for line in f:
        entry = json.loads(line.strip())
        text = entry.get("text", "")
        meta = {
            "title": entry.get("title", "Untitled"),
            "date": entry.get("date"),
            "tags": entry.get("tags", []),
            "category": entry.get("category", "uncategorized"),
            "date_friendly": entry.get("date_friendly", entry.get("date")),
        }

        # Create embedding
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        ).data[0].embedding

        # Upsert into Pinecone
        index.upsert([
            {
                "id": entry.get("id", f"journal-{meta['date']}"),
                "values": embedding,
                "metadata": {"text": text, **meta}
            }
        ])

print("✅ Journal upload completed successfully!")
