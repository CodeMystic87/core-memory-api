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


def upload_entries():
    """Upload journal entries into Pinecone index."""
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            entry = json.loads(line)

            # Extract fields safely
            text = str(entry.get("text", "")).strip()
            if not text:
                continue

            kind = entry.get("kind", "journal")
            title = str(entry.get("title", ""))
            tags = entry.get("tags", [])
            if isinstance(tags, (float, int)):
                tags = [str(tags)]

            metadata = {
                "kind": kind,
                "title": title,
                "tags": [str(tag) for tag in tags],
                "mood": str(entry.get("mood", "")),
                "people": [str(p) for p in entry.get("people", [])],
                "activities": [str(a) for a in entry.get("activities", [])],
                "keywords": [str(k) for k in entry.get("keywords", [])],
                "meta": entry.get("meta", {})
            }

            # Embed and upsert into Pinecone
            embedding = embed_text(text)
            index.upsert([
                (
                    entry.get("id", str(hash(text))),  # fallback id
                    embedding,
                    metadata
                )
            ])

    print("✅ Journal upload complete.")


if __name__ == "__main__":
    print(f"📓 Using journal file: {INPUT_FILE}")
    upload_entries()
