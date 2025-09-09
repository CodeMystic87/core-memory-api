import os
import json
import uuid
import math
from openai import OpenAI
import pinecone

print("🚀 upload_journal.py has started...")

# === Initialize OpenAI ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Initialize Pinecone ===
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("core-memory")

# === Path to your journal file ===
journal_file = "core_memory_api/journal_with_tags_and_categories.jsonl"

if not os.path.exists(journal_file):
    raise FileNotFoundError(f"❌ File not found: {journal_file}")

print(f"📖 Using journal file: {journal_file}")

# === Upload loop ===
with open(journal_file, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, start=1):
        try:
            entry = json.loads(line)

            # Skip if no valid text
            if not entry.get("text") or not isinstance(entry["text"], str):
                print(f"⚠️ Skipping line {line_num} (missing or invalid text)")
                continue

            # Guard against NaN/inf in metadata
            if "meta" in entry:
                bad_meta = False
                for k, v in entry["meta"].items():
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        print(f"⚠️ Skipping line {line_num} (bad metadata: {k}={v})")
                        bad_meta = True
                        break
                if bad_meta:
                    continue

            # Create embedding
            embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input=entry["text"]
            ).data[0].embedding

            # Upload to Pinecone
            index.upsert([
                (str(uuid.uuid4()), embedding, entry)
            ])

            print(f"✅ Uploaded line {line_num}")

        except Exception as e:
            print(f"❌ Error at line {line_num}: {e}")
