import json
import requests
from datetime import datetime

# ===== CONFIG =====
CORE_MEMORY_API = "https://core-memory-api.onrender.com/storeMemory"
INPUT_FILE = "core-memory-api/journal_with_tags_and_categories.jsonl"
VERSION = "3.7a"
TIMEZONE = "America/Phoenix"

# ===== LOAD INPUT =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"Loaded {len(entries)} entries from {INPUT_FILE}")

# ===== HELPERS =====
def normalize_date(raw_date: str):
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
        iso = dt.strftime("%Y-%m-%dT00:00:00")
        friendly = dt.strftime("%B %d, %Y")
        return iso, friendly
    except Exception:
        return None, None

def save_exception(entry, error_message):
    """Append bad entry to exceptions.log"""
    with open("exceptions.log", "a", encoding="utf-8") as log:
        log.write(json.dumps({
            "error": error_message,
            "entry": entry
        }, ensure_ascii=False) + "\n")

# ===== MIGRATION =====
migrated, skipped = 0, 0

for i, entry in enumerate(entries, 1):
    try:
        # Force text into a string
        raw_text = entry.get("text", "")
        text = str(raw_text).strip()

        if not text or len(text) < 5:
            skipped += 1
            save_exception(entry, "Too short or empty text")
            continue

        # Normalize date
        raw_date = entry.get("date") or entry.get("created_at")
        iso, friendly = normalize_date(raw_date) if raw_date else (None, None)
        if not iso:
            skipped += 1
            save_exception(entry, "Invalid or missing date")
            continue

        # Tags
        tags = [
            f"date:{iso[:10]}",
            f"date_friendly:{friendly}",
            "type:journal"
        ]

        payload = {
            "text": text,
            "kind": "journal",
            "tags": tags,
            "meta": {
                "datetime_iso": iso,
                "timezone": TIMEZONE,
                "version": VERSION
            }
        }

        # Send to API
        res = requests.post(CORE_MEMORY_API, json=payload)
        if res.status_code == 200:
            migrated += 1
            print(f"✅ Migrated {i}/{len(entries)}")
        else:
            skipped += 1
            save_exception(entry, f"API error {res.status_code}: {res.text}")

    except Exception as e:
        skipped += 1
        save_exception(entry, f"Exception: {str(e)}")

print(f"\nMigration complete ✅ Migrated {migrated}, Skipped {skipped}")
print("Check exceptions.log for skipped entries.")
