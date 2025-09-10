import json
import requests
import time
from datetime import datetime

# =========================
# CONFIG
# =========================
CORE_MEMORY_API = "https://core-memory-api.onrender.com/storeMemory"
TIMEZONE = "America/Phoenix"
VERSION = "v3.7a"
INPUT_FILE = "journal_with_tags_and_categories.jsonl"
LOG_FILE = "migration_log.txt"

# =========================
# LOAD ENTRIES
# =========================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"Loaded {len(entries)} entries from {INPUT_FILE}")


# =========================
# FORMAT ENTRY
# =========================
def format_entry(entry):
    try:
        # --- Date handling ---
        date_raw = entry.get("date", "")
        if not date_raw:
            return None
        try:
            dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
        except Exception:
            return None
        date_str = dt.date().isoformat()

        # --- Tags (baseline schema) ---
        tags = [
            f"date:{date_str}",
            f"date_friendly:{dt.strftime('%B %d, %Y')}",
            "type:journal",
        ]
        if "category" in entry and entry["category"]:
            tags.append(f"category:{entry['category']}")
        if "tags" in entry and entry["tags"]:
            tags.extend(entry["tags"])

        # --- Text ---
        text = str(entry.get("text", "")).strip()
        if not text or len(text) < 20:  # skip junk
            return None

        # --- Final payload in CoreMemory schema ---
        return {
            "text": text,
            "kind": "journal",
            "tags": tags,
            "meta": {
                "datetime_iso": dt.isoformat(),
                "timezone": TIMEZONE,
                "version": VERSION,
            },
        }
    except Exception as e:
        print(f"⚠️ Skipped due to formatting error: {e}")
        return None


# =========================
# UPLOAD ENTRY WITH RETRIES
# =========================
def upload_entry(payload, retries=3):
    for attempt in range(1, retries + 1):
        try:
            res = requests.post(CORE_MEMORY_API, json=payload, timeout=10)
            if res.status_code == 200:
                return True
            else:
                print(f"❌ Failed {res.status_code}: {res.text}")
        except Exception as e:
            print(f"⚠️ Error attempt {attempt}: {e}")
        time.sleep(2 * attempt)  # backoff
    return False


# =========================
# MIGRATION LOOP
# =========================
with open(LOG_FILE, "w", encoding="utf-8") as log:
    for i, entry in enumerate(entries, 1):
        payload = format_entry(entry)
        if not payload:
            log.write(f"[SKIPPED] {i} — invalid or too short\n")
            continue

        success = upload_entry(payload)
        if success:
            log.write(f"[SAVED]   {i} — {payload['tags'][0]}\n")
            print(f"✅ Saved {payload['tags'][0]} — {payload['text'][:40]}...")
        else:
            log.write(f"[FAILED]  {i} — {payload['tags'][0]}\n")

print("🎉 Migration finished. Check migration_log.txt for details.")
