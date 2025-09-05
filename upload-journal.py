import json
import os
from openai import OpenAI
import pinecone

# Load API keys (make sure these are set in Render env vars)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV", "us-east-1")
)

# Connect to Pinecone index
index = pinecone.Index("dayonememories")

# Load JSONL file
print("🔍 Loading journal file...")
with open("journal_with_tags_and_categories.jsonl", "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"✅ Loaded {len(entries)} journal entries")

# Upload in batches
batch_size = 10
for i in range(0, len(entries), batch_size):
    batch = entries[i:i + batch_size]
    texts = [entry["text"] for entry in batch]

    # Create embeddings
    print(f"🧠 Creating embeddings for batch {i//batch_size + 1}...")
    embeddings = openai.embeddings.create(
        model="text-embedding-3-small", 
        input=texts
    )

    # Push to Pinecone
    vectors = []
    for j, entry in enumerate(batch):
        vectors.append({
            "id": f"entry-{i+j}",
            "values": embeddings.data[j].embedding,
            "metadata": {
                "tag": entry["tag"],
                "category": entry["category"],
                "text": entry["text"]
            }
        })
    index.upsert(vectors)

    print(f"✅ Uploaded batch {i//batch_size + 1} ({len(batch)} records)")
