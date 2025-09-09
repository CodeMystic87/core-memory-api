import os
import json
from openai import OpenAI
import pinecone

# === Config ===
INPUT_FILE = "core_memory_api/journal_fixed.jsonl"   # <- updated to fixed journal file
INDEX_NAME = "core-memory"

# === Initialize OpenAI ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Initialize Pinecone ===
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(INDEX_NAME)

def upload_entries():
    print(f"📂 Using journal file: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for i, line in enumerate(infile, start=1):
            entry = json.loads(line)

            # Ensure every entry has a kind (fallback if migration missed one)
            if "kind" not in entry:
                entry["kind"] = "journal"

            text = entry.get("text", "")
            if not text.strip():
                continue  # skip empty entries

            # Create embedding
            embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding

            # Upsert into Pinecone
            index.upsert([
                (
                    str(i),  # unique ID
                    embedding,
                    {"text": text, "tags": entry.get("tags", []), "kind": entry.get("kind", "journal")}
                )
            ])

            if i % 50 == 0:
                print(f"✅ Uploaded {i} entries...")

    print("🎉 Upload complete!")

if __name__ == "__main__":
    upload_entries()
