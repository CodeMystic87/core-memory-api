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
    """Ensure Pinecone metadata is valid: only str, number, bool, or list[str]."""
    if not meta or not isinstance(meta, dict):
        return {}
    cleaned = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)  # fallback: stringify
    return cleaned


def upload_entries():
    print(f"✅ Using journal file: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            entry = json.loads(line)

            vector = embed_text(entry["text"])
            metadata = clean_metadata(entry.get("meta", {}))
            metadata["kind"] = entry.get("kind", "journal")  # always ensure kind is present

            index.upsert([
                {
                    "id": entry.get("id", entry["meta"].get("datetime_iso", "")),
                    "values": vector,
                    "metadata": metadata
                }
            ])
    print("🎉 Upload complete.")


if __name__ == "__main__":
    upload_entries()
