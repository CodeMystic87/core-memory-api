import os
import json
import numpy as np
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
    """Recursively clean metadata (remove NaN, inf, None)."""
    if isinstance(meta, dict):
        return {k: clean_metadata(v) for k, v in meta.items()}
    elif isinstance(meta, list):
        return [clean_metadata(v) for v in meta]
    elif isinstance(meta, float):
        if np.isnan(meta) or np.isinf(meta):
            return None
        return float(meta)
    elif meta is None:
        return None
    return meta


def clean_vector(vec):
    """Replace NaN / inf values in embedding vector with 0.0"""
    return [
        0.0 if (v is None or np.isnan(v) or np.isinf(v)) else float(v)
        for v in vec
    ]


def upload_entries():
    print(f"✅ Using journal file: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue

            entry = json.loads(line)
            vector = embed_text(entry["text"])
            vector = clean_vector(vector)

            metadata = clean_metadata(entry.get("meta", {}))
            metadata["kind"] = entry.get("kind", "journal")  # ensure kind always exists

            index.upsert([{
                "id": entry.get("id", entry["meta"].get("datetime_iso", "")),
                "values": vector,
                "metadata": metadata
            }])

    print("🚀 Upload complete.")


if __name__ == "__main__":
    upload_entries()
