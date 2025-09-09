import os
import json
from openai import OpenAI
import pinecone

print("🚀 upload_journal.py has started...")

# === Initialize OpenAI ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Initialize Pinecone ===
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

# === Load Journal File ===
file_path = "core_memory_api/journal_with_tags_and_categories.jsonl"
print(f"📂 Using journal file: {file_path}")

with open(file_path, "r") as f:
    for i, line in enumerate(f, 1):
        try:
            entry = json.loads(line)
            text = entry.get("text", "").strip()

            if not text:
                print(f"⚠️ Skipping empty text entry at line {i}")
                continue

            # Generate embedding safely
            try:
                embedding = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                ).data[0].embedding
            except Exception as e:
                print(f"⚠️ Failed to embed entry at line {i}, skipping. Error: {e}")
                continue

            # Insert into Pinecone
            index.upsert([{
                "id": f"journal-{i}",
                "values": embedding,
                "metadata": entry
            }])

            if i % 50 == 0:
                print(f"✅ Processed {i} entries...")

        except Exception as e:
            print(f"⚠️ Failed to process line {i}, skipping. Error: {e}")

print("🎉 Upload complete! Check Pinecone index for results.")
