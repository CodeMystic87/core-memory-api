import json
import requests
from datetime import datetime
import pytz
import os

# ===== CONFIG =====
CORE_MEMORY_API = "https://core-memory-api.onrender.com/storeMemory"
INPUT_FILE = "core-memory-api/journal_with_tags_and_categories.jsonl"
TIMEZONE = "America/Phoenix"
VERSION = "3.7a"

# ===== LOGGING =====
exceptions_log = open("exceptions.log", "w", encoding="utf-8")

def log_exception(entry, reason):
    json.dump({"error": reason, "entry": entry}, exceptions_log, ensure_ascii=False)
    exceptions_log.write("\n")

# ===== HELPERS =====
def normalize_date(raw_date: str) -> str:
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "1970-01-01"  # fallback if missing/invalid

def make_friendly_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%B %-d, %Y")  # e.g. September 10, 2025
    except Exception:
        return "January 1, 1970"

def ensure_entry(entry):
    """Self-heal entry structure if missing fields."""
    healed = {}

    # 1. Text
    text = entry.get("text")
    if not text or len(text.strip()) == 0:
        raise ValueError("Missing text")

    healed["text"] = text.strip()

    # 2. Kind
    kind = entry.get("kind", "note")
    if kind not in ["journal", "note", "task", "idea", "milestone",
                    "conversation", "decision", "goal", "reference"]:
        kind = "note"
    healed["kind"] = kind

    # 3. Meta
    meta = entry.get("meta", {})
    dt_iso = meta.get("datetime_iso")
    if not dt_iso:
        dt_iso = datetime.now(pytz.UTC).isoformat()
    timezone = meta.get("timezone", TIMEZONE)
    version = meta.get("version", VERSION)

    healed["meta"] = {
        "datetime_iso": dt_iso,
        "timezone": timezone,
        "version": version,
    }

    # 4. Tags
    tags = entry.get("tags", [])
    date_tag = None
    for t in tags:
        if t.startswith("date:"):
            date_tag = t.split(":", 1)[1]

    if not date_tag:
        # fallback to normalized date from meta if possible
        try:
            date_str = dt_iso.split("T")[0]
        except Exception:
            date_str = "1970-01-01"
        date_tag = normalize_date(date_str)

    friendly = make_friendly_date(date_tag)
    base_tags = [
        f"date:{date_tag}",
        f"date_friendly:{friendly}",
        f"type:{kind}"
    ]
    healed["tags"] = list(set(tags + base_tags))

    # 5. Optional extras
    for field in ["mood", "people", "activities", "keywords"]:
        if field in entry:
            healed[field] = entry[field]

    return healed


# ===== MIGRATION =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    entries = [json.loads(line) for line in f]

print(f"Loaded {len(entries)} entries from {INPUT_FILE}")

success_count = 0
skip_count = 0

for idx, entry in enumerate(entries, 1):
    try:
        healed_entry = ensure_entry(entry)

        res = requests.post(CORE_MEMORY_API, json=healed_entry)
        if res.status_code == 200:
            success_count += 1
            if success_count % 100 == 0:
                print(f"✅ Migrated {success_count}/{len(entries)} so far…")
        else:
            log_exception(entry, f"API error {res.status_code}")
            skip_count += 1

    except Exception as e:
        log_exception(entry, str(e))
        skip_count += 1

exceptions_log.close()

print(f"\nMigration finished!")
print(f"✅ Successful: {success_count}")
print(f"⚠️ Skipped: {skip_count}")
print("Check exceptions.log for details on skipped entries.") 
