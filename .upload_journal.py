import os
import json
import pinecone
from openai import OpenAI

print("🚀 upload_journal.py has started running...")

# --- Initialize OpenAI ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Initialize Pinecone ---
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "dayonememories"

# Check if index exists, create if missing
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # text-embedding-3-small dimension
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENV"))
    )
else:
    print(f"✅ Pinecone index '{index_name}' already exists.")

index = pc.Index(index_name)

# --- Load Journal Entries ---
file_path = "journal_with_tags_and_categories.jsonl"

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit(1)

with open(file_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"📓 Loaded {len(entries)} journal entries")

# --- Process in Batches ---
batch_size = 100

for i in range(0, len(entries), batch_size):
    batch = entries[i:i + batch_size]
    texts = [str(entry["text"]) for entry in batch]

    print(f"🔄 Processing batch {i // batch_size + 1} with {len(texts)} entries...")

    # Create embeddings
    try:
        embeddings = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        break

    # Format for Pinecone
    vectors = [
        {
            "id": f"entry-{i+j}",
            "values": embeddings.data[j].embedding,
            "metadata": {
                "tag": entry.get("tag", ""),
                "category": entry.get("category", ""),
                "text": entry["text"]
            }
        }
        for j, entry in enumerate(batch)
    ]

    # Upload batch
    try:
        index.upsert(vectors)
        print(f"✅ Uploaded batch {i // batch_size + 1}")
    except Exception as e:
        print(f"❌ Error uploading to Pinecone: {e}")
        break

print("🎉 Finished uploading all journal entries to Pinecone!")
