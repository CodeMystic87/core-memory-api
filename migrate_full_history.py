import json
import requests
from datetime import datetime
import pytz
import os

# ===== CONFIG =====
CORE_MEMORY_API = "https://core-memory-api.onrender.com/storeMemory"
INPUT_FILE = "journal_with_tags_and_categories.jsonl"
TIMEZONE = "America/Phoenix"
VERSION = "3.7b"

# ===== HELPERS =====
def normalize_date(raw_date: str):
    """Convert input date string into ISO format + friendly format."""
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
    except Exception:
        dt = datetime.now(pytz.timezone(TIMEZONE))  # fallback to now
    return dt.isoformat(), dt.strftime("%B %d, %Y")

def build_payload(entry):
    """Builds the CoreMemory payload with schema v3.7b."""
    text = entry.get("text", "").strip()
    if not text:
        return None  # skip empty

    kind = entry.get("kind", "note")
    tags = entry.get("tags", [])

    # Normalize dates
    raw_date = entry.get("date", None)
    if raw_date:
        iso_date, friendly_date = normalize_date(raw_date)
        tags.append(f"date:{raw_date}")
        tags.append(f"date_friendly:{friendly_date}")
    else:
        iso_date, friendly_date = normalize_date(datetime.now().strftime("%Y-%m-%d"))

    tags.append(f"type:{kind}")

    payload = {
        "text": text,
        "kind": kind,
        "tags": list(set(tags)),  # deduplicate
        "mood": entry.get("mood"),
        "people": entry.get("people", []),
        "activities": entry.get("activities", []),
        "keywords": entry.get("keywords", []),
        "meta": {
            "datetime_iso": iso_date,
            "timezone": TIMEZONE,
            "version": VERSION
        }
    }
    return payload

# ===== MAIN MIGRATION =====
exceptions = []
total = 0
success = 0

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"❌ Input file not found: {INPUT_FILE}")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"📥 Loaded {len(entries)} entries from {INPUT_FILE}")

for idx, entry in enumerate(entries, 1):
    total += 1
    payload = build_payload(entry)

    if not payload:
        exceptions.append({"entry": entry, "error": "Empty or invalid text"})
        print(f"⚠️ Skipped entry {idx}/{len(entries)} (empty text)")
        continue

    try:
        res = requests.post(CORE_MEMORY_API, json=payload)
        if res.status_code == 200:
            success += 1
            print(f"✅ Migrated {idx}/{len(entries)} ({success} successful)")
        else:
            exceptions.append({"entry": entry, "error": f"API {res.status_code}"})
            print(f"❌ Failed {idx}/{len(entries)} - API error {res.status_code}")
    except Exception as e:
        exceptions.append({"entry": entry, "error": str(e)})
        print(f"❌ Exception on entry {idx}/{len(entries)}: {str(e)}")

# ===== SUMMARY =====
print("\n========================")
print(f"🎉 Migration complete")
print(f"✅ Successful: {success}")
print(f"⚠️ Failed: {len(exceptions)}")
print("========================")

if exceptions:
    with open("exceptions.log", "w", encoding="utf-8") as log:
        json.dump(exceptions, log, indent=2, ensure_ascii=False)
    print("⚠️ Some entries failed — logged to exceptions.log")
else:
    print("🎉 All entries migrated successfully — no errors!")
