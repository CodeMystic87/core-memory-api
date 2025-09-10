import json
import requests
from datetime import datetime

CORE_MEMORY_API = "https://core-memory-api.onrender.com/storeMemory"
INPUT_FILE = "journal_with_tags_and_categories.jsonl"
VERSION = "3.7a"
TIMEZONE = "America/Phoenix"

# load input
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]
print(f"📥 Loaded {len(entries)} entries")

def normalize_date(raw_date):
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
        return dt.isoformat(), dt.strftime("%B %d, %Y")
    except Exception:
        now = datetime.now()
        return now.isoformat(), now.strftime("%B %d, %Y")

for idx, entry in enumerate(entries, 1):
    text = entry.get("text", "").strip()
    if not text or len(text) < 5:
        print(f"⏭️ Skipped {idx}/{len(entries)} (empty/short)")
        continue

    raw_date = entry.get("date", "")
    iso, friendly = normalize_date(raw_date)

    new_entry = {
        "kind": "journal",
        "text": text,
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
        r = requests.post(CORE_MEMORY_API, json=new_entry)
        if r.status_code == 200:
            print(f"✅ Migrated {idx}/{len(entries)} ({raw_date})")
        else:
            print(f"❌ Error {r.status_code} on {idx}: {r.text}")
    except Exception as e:
        print(f"💥 Exception {idx}: {e}")
