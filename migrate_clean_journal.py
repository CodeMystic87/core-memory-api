import json
import math
from datetime import datetime

INPUT_FILE = "journal_with_tags_and_categories.jsonl"
OUTPUT_FILE = "journal_fixed.jsonl"

def normalize_date(value):
    """Try to coerce dates into ISO8601 (YYYY-MM-DDTHH:MM:SSZ)."""
    try:
        # Already ISO
        datetime.fromisoformat(value.replace("Z", ""))
        return value
    except:
        try:
            # Try parsing simple YYYY-MM-DD
            return datetime.strptime(value, "%Y-%m-%d").isoformat() + "Z"
        except:
            return None

def clean_entry(entry):
    # Always enforce kind
    entry["kind"] = entry.get("kind", "journal")

    # Ensure text exists
    if not entry.get("text") or not entry["text"].strip():
        return None

    # Fix meta
    meta = entry.get("meta", {})
    if not isinstance(meta, dict) or len(meta) == 0:
        entry["meta"] = None
    else:
        fixed_meta = {}
        if "datetime_iso" in meta:
            fixed_meta["datetime_iso"] = normalize_date(meta["datetime_iso"])
        if "timezone" in meta:
            fixed_meta["timezone"] = str(meta["timezone"])
        if "version" in meta:
            fixed_meta["version"] = str(meta["version"])
        entry["meta"] = fixed_meta if fixed_meta else None

    # Force lists
    for key in ["tags", "people", "activities", "keywords"]:
        if not isinstance(entry.get(key), list):
            entry[key] = []

    # Mood should always be a string
    if "mood" in entry and entry["mood"] is None:
        entry["mood"] = ""

    # Remove NaN/Infinity
    for k, v in list(entry.items()):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            entry[k] = None

    return entry

def migrate():
    count_in, count_out = 0, 0
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for line in infile:
            count_in += 1
            try:
                entry = json.loads(line)
                clean = clean_entry(entry)
                if clean:
                    outfile.write(json.dumps(clean, ensure_ascii=False) + "\n")
                    count_out += 1
            except Exception as e:
                print(f"⚠️ Skipped line {count_in}: {e}")
    print(f"✅ Migration complete. {count_out}/{count_in} entries saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    migrate()
