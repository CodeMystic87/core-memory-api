import json
import requests
import time
from datetime import datetime

# =========================
# CONFIG
# =========================
CORE_MEMORY_API = "https://core-memory-api.onrender.com/storeMemory"
TIMEZONE = "America/Phoenix"
VERSION = "v1"

# =========================
# LOAD JOURNAL JSONL
# =========================
INPUT_FILE = "journal_with_tags_and_categories.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"Loaded {len(entries)} entries from {INPUT_FILE}")

# =========================
# HELPER: format entry → CoreMemory schema
# =========================
def format_entry(entry):
    try:
        date_str = entry.get("date")[:10]  # YYYY-MM-DD
        date_obj = datetime.fromisoformat(date_str)

        tags = [
            f"date:{date_str}",
            f"date_friendly:{date_obj.strftime('%B %-d, %Y')}",
            "type:journal",
        ]

        if "category" in entry and entry["category"]:
            tags.append(f"category:{entry['category']}")
        if "tags" in entry and entry["tags"]:
            tags.extend(entry["tags"])

        # Always coerce text to string
        text = str(entry.get("text", "")).strip()

        # Skip empty or very short entries
        if not text or len(text) < 20:
            print(f"⏩ Skipped {tags[0]} — too short or empty")
            return None

        return {
            "text": text,
            "kind": "journal",
            "tags": tags,
            "meta": {
                "datetime_iso": entry.get("date"),
                "timezone": TIMEZONE,
                "version": VERSION,
            }
        }
    except Exception as e:
        print(f"⚠️ Skipped entry due to formatting error: {e}")
        return None

# =========================
# UPLOAD (batch, safe)
# =========================
def upload_entries(batch):
    for e in batch:
        payload = format_entry(e)
        if not payload:
            continue
        try:
            res = requests.post(CORE_MEMORY_API, json=payload)
            if res.status_code == 200:
                print(f"✅ Saved {payload['tags'][0]} — {payload['text'][:40]}...")
            else:
                print(f"❌ Error {res.status_code}: {res.text}")
        except Exception as ex:
            print(f"⚠️ Failed to upload: {ex}")

# =========================
# RUN IN BATCHES
# =========================
BATCH_SIZE = 10   # start small; once stable, increase (e.g., 100)

for i in range(0, len(entries), BATCH_SIZE):
    batch = entries[i:i+BATCH_SIZE]
    print(f"\nUploading entries {i+1} → {i+len(batch)}...")
    upload_entries(batch)
    time.sleep(0.5)  # safety delay between batches

print("🎉 Migration complete")
