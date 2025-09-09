import os
import json
from openai import OpenAI
import pinecone

# === Setup API clients ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

# === Detect which journal file to use ===
FIXED_FILE = "journal_fixed.jsonl"
ORIGINAL_FILE = "journal_with_tags_and_categories.jsonl"

if os.path.exists(FIXED_FILE):
    INPUT_FILE = FIXED_FILE
    print(f"📂 Using FIXED journal file: {INPUT_FILE}")
elif os.path.exists(ORIGINAL_FILE):
    INPUT_FILE = ORIGINAL_FILE
    print(f"📂 Using ORIGINAL journal file: {INPUT_FILE}")
else:
    raise FileNotFoundError("❌ No journal file found!")

# === Upload entries ===
print(f"🚀 Upload starting with file: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, start=1):
        try:
            entry = json.loads(line)

            # Validate text
            if not entry.get("text") or not entry["text"].strip():
                print(f"⚠️ Skipping empty entry at line {i}")
                continue

            # Add embeddings
            embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input=entry["text"]
            ).data[0].embedding

            # Upsert into Pinecone
            index.upsert([
                (
                    f"journal-{i}",
                    embedding,
                    entry
                )
            ])

            print(f"✅ Uploaded entry {i}")

        except Exception as e:
            print(f"❌ Error on line {i}: {e}")

print("🎉 Upload complete!")
