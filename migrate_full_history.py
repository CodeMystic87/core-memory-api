import json
import requests
from datetime import datetime

# ===== CONFIG =====
CORE_MEMORY_API = "https://core-memory-api.onrender.com/storeMemory"
INPUT_FILE = "core-memory-api/Journal_with_tags_and_categories.jsonl"
VERSION = "3.7a"
TIMEZONE = "America/Phoenix"

# ===== LOAD INPUT =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"Loaded {len(entries)} entries from {INPUT_FILE}")

# ===== HELPERS =====
def normalize_date(raw_date: str):
    """
    Convert input date string into ISO + friendly formats.
    Expects YYYY-MM-DD already, but fallback if missing.
    """
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
    except Exception:
        # fallback default if no valid date
        return "2000-01-01", "January 1, 2000"

    iso = dt.strftime("%Y-%m-%d")
    friendly = dt.strftime("%B %-d, %Y")  # e.g., November 12, 2019
    return iso, friendly

def build_payload(entry):
    """
    Force every entry to include:
      - kind
      - tags: date + friendly + type
      - meta
    """
    raw_date = entry.get("date") or entry.get("tags", [""])[0].replace("date:", "")
    date_iso, date_friendly = normalize_date(raw_date)

    payload = {
        "text": entry.get("text", "").strip(),
        "kind": "journal",  # force journal
        "tags": [
            f"date:{date_iso}",
            f"date_friendly:{date_friendly}",
            "type:journal"
        ],
        "meta": {
            "datetime_iso": f"{date_iso}T00:00:00",
            "timezone": TIMEZONE,
            "version": VERSION
        }
    }

    return payload

def upload_entry(payload):
    try:
        res = requests.post(CORE_MEMORY_API, json=payload, timeout=10)
        if res.status_code == 200:
            return True
        else:
            print(f"⚠️ Failed: {res.status_code}, {res.text}")
            return False
    except Exception as e:
        print(f"⚠️ Exception while uploading: {e}")
        return False

# ===== MAIN LOOP =====
success, skipped = 0, 0

for i, entry in enumerate(entries, start=1):
    payload = build_payload(entry)

    # Skip empty/short text
    if not payload["text"] or len(payload["text"]) < 15:
        skipped += 1
        print(f"⏭️ Skipped {i} — too short/empty")
        continue

    if upload_entry(payload):
        success += 1
        print(f"✅ Saved {i}/{len(entries)} — {payload['tags'][0]}: {payload['text'][:40]}...")

print(f"\nDone. Uploaded {success}, skipped {skipped}.")
