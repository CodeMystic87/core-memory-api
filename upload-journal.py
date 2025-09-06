import os
import json
import pinecone
from openai import OpenAI

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")  # should match "us-east1-gcp" etc.
)

index = pinecone.Index("dayonememories")

# Path to your JSONL file
file_path = "journal_with_tags_and_categories.jsonl"

print("🔍 Checking for journal file...")
if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

print("✅ Found journal file!")

# Load entries
with open(file_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"📓 Loaded {len(entries)} entries from journal.")

# Batch settings
batch_size = 10

for i in range(0, len(entries), batch_size):
    batch = entries[i:i + batch_size]
    texts = [entry["text"] for entry in batch]

    print(f"➡️ Processing batch {i//batch_size + 1} with {len(texts)} entries...")

    # Create embeddings
    try:
        embeddings = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        ).data
        print(f"✅ Created embeddings for {len(embeddings)} entries.")
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        continue

    # Prepare Pinecone vectors
    vectors = []
    for j, entry in enumerate(batch):
        vectors.append((
            entry["id"],  # unique ID for Pinecone
            embeddings[j].embedding,
            {
                "tags": entry.get("tags", []),
                "category": entry.get("category", ""),
                "text": entry.get("text", "")
            }
        ))

    # Upsert into Pinecone
    try:
        index.upsert(vectors=vectors)
        print(f"✅ Uploaded {len(vectors)} vectors to Pinecone.")
    except Exception as e:
        print(f"❌ Error uploading to Pinecone: {e}")

print("🎉 Finished uploading all journal entries.")
