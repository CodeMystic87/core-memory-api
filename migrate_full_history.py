import json
import requests
from datetime import datetime

# ===== CONFIG =====
CORE_MEMORY_API = "https://core-memory-api.onrender.com/storeMemory"
INPUT_FILE = "journal_with_tags_and_categories.jsonl"
VERSION = "3.7a"
TIMEZONE = "America/Phoenix"

# ===== LOAD INPUT =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"📥 Loaded {len(entries)} entries from {INPUT_FILE}")

# ===== HELPERS =====
def normalize_date(raw_date: str):
    """Convert input date into ISO + friendly formats."""
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
        iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
        friendly = dt.strftime("%B %d, %Y")
        return iso, friendly
    except Exception:
        now = datetime.now()
        return now.isoformat(), now.strftime("%B %d, %Y")

# ===== MIGRATION LOOP =====
for idx, entry in enumerate(entries, 1):
    raw_text = entry.get("text", "").strip()
    raw_date = entry.get("date", "")

    # Skip empty or too-short entries
    if not raw_text or len(raw_text) < 5:
        print(f"⏭️ Skipped {idx}/{len(entries)} (empty/too short)")
        continue

    iso, friendly = normalize_date(raw_date)

    # Build new normalized entry
    new_entry = {
        "kind": "journal",
        "text": raw_text,
        "tags": [
            f"date:{raw_date}",
            f"date_friendly:{friendly}",
            "type:journal"
        ],
        "meta": {
            "datetime_iso": iso,
            "timezone": TIMEZONE,
            "version": VERSION
        }
    }

    try:
        res = requests.post(CORE_MEMORY_API, json=new_entry)
        if res.status_code == 200:
            print(f"✅ Migrated {idx}/{len(entries)} ({raw_date})")
        else:
            print(f"❌ Error {res.status_code} on {idx}/{len(entries)}: {res.text}")
    except Exception as e:
        print(f"💥 Exception on {idx}/{len(entries)}: {e}")
