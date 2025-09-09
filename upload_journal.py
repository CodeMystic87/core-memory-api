import os
import json
from openai import OpenAI
import pinecone

# Input file – make sure this matches your migrated journal
INPUT_FILE = "core_memory_api/journal_fixed.jsonl"

# Initialize clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

def embed_text(text):
    """Generate embeddings safely from OpenAI."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def clean_metadata(meta):
    """Recursively clean metadata to remove NaN/Infinity/-Infinity."""
    if isinstance(meta, dict):
        return {k: clean_metadata(v) for k, v in meta.items()}
    elif isinstance(meta, list):
        return [clean_metadata(v) for v in meta]
    elif isinstance(meta, float):
        if meta != meta or meta in (float("inf"), float("-inf")):
            return None
        return meta
    return meta

def upload_entries():
    print(f"📖 Using journal file: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue

            entry = json.loads(line)

            vector = embed_text(entry.get("text", ""))

            # Always ensure metadata is safe
            meta = entry.get("meta", {})
            metadata = clean_metadata(meta)
            metadata["kind"] = entry.get("kind", "journal")

            index.upsert([{
                "id": entry.get("id", meta.get("datetime_iso", "")),
                "values": vector,
                "metadata": metadata
            }])

    print("✅ Upload complete.")

if __name__ == "__main__":
    upload_entries()
