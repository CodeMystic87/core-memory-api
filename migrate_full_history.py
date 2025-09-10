import json
import requests
from datetime import datetime
import os
import glob

# ===== CONFIG =====
CORE_MEMORY_API = "https://core-memory-api.onrender.com/storeMemory"
INPUT_FILE = "core-memory-api/journal_with_tags_and_categories.jsonl"
VERSION = "3.7a"
TIMEZONE = "America/Phoenix"

# ===== AUTO-FIX INPUT FILENAME =====
if not os.path.exists(INPUT_FILE):
    matches = glob.glob("journal_with_tags*.jsonl")
    if matches:
        os.rename(matches[0], INPUT_FILE)
        print(f"✅ Auto-renamed {matches[0]} -> {INPUT_FILE}")
    else:
        raise FileNotFoundError("❌ Could not find journal_with_tags JSONL file")

# ===== LOAD INPUT =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"📥 Loaded {len(entries)} entries from {INPUT_FILE}")

# ===== HELPERS =====
def normalize_date(raw_date: str):
    """
    Convert input date string into ISO + friendly formats.
    Expects YYYY-MM-DD, falls back if missing.
    """
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
    except Exception:
        dt = datetime(2000, 1, 1)  # fallback default
    return {
        "iso": dt.strftime("%Y-%m-%dT00:00:00"),
        "friendly": dt.strftime("%B %d, %Y")
    }

def build_payload(entry):
    """Standardize entry into CoreMemory schema with tags + metadata."""
    raw_date = entry.get("date") or "2000-01-01"
    norm = normalize_date(raw_date)

    text = entry.get("text", "").strip()
    if not text or len(text) < 5:
        return None  # skip empty/too short

    return {
        "text": text,
        "kind": "journal",
        "tags": [
            f"date:{raw_date}",
            f"date_friendly:{norm['friendly']}",
            "type:journal"
        ],
        "meta": {
            "datetime_iso": norm["iso"],
            "timezone": TIMEZONE,
            "version": VERSION
        }
    }

# ===== UPLOAD =====
success, skipped, failed = 0, 0, 0
for i, e in enumerate(entries, 1):
    payload = build_payload(e)
    if not payload:
        print(f"⏭️ Skipped entry {i} (empty/too short)")
        skipped += 1
        continue

    try:
        r = requests.post(CORE_MEMORY_API, json=payload, timeout=10)
        if r.status_code == 200:
            print(f"✅ Saved {i}: {payload['tags'][0]} ... {payload['text'][:40]}...")
            success += 1
        else:
            print(f"❌ Failed {i}: {r.status_code} {r.text}")
            failed += 1
    except Exception as ex:
        print(f"🔥 Error on {i}: {ex}")
        failed += 1

print("\n==== Migration Summary ====")
print(f"✅ {success} saved")
print(f"⏭️ {skipped} skipped")
print(f"❌ {failed} failed")
