from openai import OpenAI
import pinecone
import os
import json

print("🚀 upload_journal.py has started running...")

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("dayonememories")

# Load your journal JSONL file
file_path = "journal_with_tags_and_categories.jsonl"
with open(file_path, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"📖 Loaded {len(entries)} journal entries")

# Batch upload
batch_size = 100
for i in range(0, len(entries), batch_size):
    batch = entries[i:i + batch_size]
    texts = [entry["text"] for entry in batch]
    ids = [str(entry["id"]) for entry in batch]

    # Get embeddings (NEW format)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    vectors = [
        {"id": ids[j], "values": response.data[j].embedding, "metadata": batch[j]}
        for j in range(len(batch))
    ]

    # Upload to Pinecone
    index.upsert(vectors)
    print(f"✅ Processed batch {i // batch_size + 1} with {len(batch)} entries")

print("🎉 Finished uploading all entries to Pinecone!")
