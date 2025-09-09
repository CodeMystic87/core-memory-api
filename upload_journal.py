import os
import json
from openai import OpenAI
import pinecone

print("🚀 upload_journal.py has started running...")

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

# Path to your JSONL file
file_path = "journal_with_tags_and_categories.jsonl"

print("📂 Checking for journal file...")
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

# Load entries
with open(file_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"✅ Loaded {len(entries)} journal entries")

# Process in batches
batch_size = 100
uploaded_count = 0

for i in range(0, len(entries), batch_size):
    batch = entries[i:i + batch_size]
    vectors = []

    for j, entry in enumerate(batch):
        try:
            # Safely coerce text to string
            raw_text = entry.get("text", "")
            text = str(raw_text).strip()

            # Skip empty or invalid text
            if not text:
                print(f"⚠️ Skipping empty text entry: {entry}")
                continue

            # Create embedding
            embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            ).data[0].embedding

            # Build Pinecone vector
            vectors.append({
                "id": str(entry.get("id", f"entry-{i}-{j}")),  # unique fallback
                "values": embedding,
                "metadata": {
                    "text": text,
                    "title": entry.get("title", "Untitled"),
                    "tags": entry.get("tags", []),
                    "category": entry.get("category", "uncategorized"),
                    "date": entry.get("date", ""),
                    "date_friendly": entry.get("date_friendly", "")
                }
            })

        except Exception as e:
            print(f"❌ Skipping entry due to error: {e}, entry: {entry}")
            continue

    if vectors:
        index.upsert(vectors)
        uploaded_count += len(vectors)
        print(f"✅ Uploaded {len(vectors)} entries in batch {i // batch_size + 1}")
    else:
        print(f"⚠️ No valid entries in batch {i // batch_size + 1}")

print(f"\n🎉 Finished uploading! Total successful entries: {uploaded_count}/{len(entries)}")
