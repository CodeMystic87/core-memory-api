import os
import json
import math
from openai import OpenAI
import pinecone

# Input file - make sure this matches your migrated journal
INPUT_FILE = "core_memory_api/journal_fixed.jsonl"

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

def clean_metadata(meta):
    """Recursively replace NaN, Infinity, and -Infinity with None"""
    if isinstance(meta, dict):
        return {k: clean_metadata(v) for k, v in meta.items()}
    elif isinstance(meta, list):
        return [clean_metadata(v) for v in meta]
    elif isinstance(meta, float):
        if math.isnan(meta) or math.isinf(meta):
            return None
        return meta
    else:
        return meta

def embed_text(text):
    """Generate embeddings safely from OpenAI"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def upload_entries():
    print(f"📖 Using journal file: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            entry = json.loads(line)

            vector = embed_text(entry.get("text", ""))
            metadata = clean_metadata(entry.get("meta", {}))
            metadata["kind"] = entry.get("kind", "journal")  # ensure kind is present

            # Always provide a stable unique ID
            vector_id = entry.get("id") or entry["meta"].get("datetime_iso", "unknown")

            index.upsert([{
                "id": vector_id,
                "values": vector,
                "metadata": metadata
            }])

    print("✅ Upload complete.")

if __name__ == "__main__":
    upload_entries()
