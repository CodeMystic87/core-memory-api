import json
import requests
from datetime import datetime
import pytz

API_URL = "https://core-memory-api.onrender.com/storeMemory"
TIMEZONE = "America/Phoenix"
VERSION = "3.7a"

def normalize_entry(raw_entry):
    """Convert a raw JSONL entry into CoreMemory schema."""
    try:
        # Extract text
        text = raw_entry.get("text", "").strip()
        if not text or len(text) < 5:  # too short = skip
            return None, "empty_or_short"

        # Extract kind, default to note
        kind = raw_entry.get("kind", "note")
        allowed_kinds = {
            "journal", "note", "task", "idea",
            "milestone", "conversation", "decision",
            "goal", "reference"
        }
        if kind not in allowed_kinds:
            # fallback to note but log
            kind = "note"
            warning = "invalid_kind"
        else:
            warning = None

        # Extract date → meta
        date_str = raw_entry.get("date") or raw_entry.get("created") or None
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            except Exception:
                dt = datetime.utcnow()
        else:
            dt = datetime.utcnow()

        dt_local = dt.astimezone(pytz.timezone(TIMEZONE))
        date_iso = dt_local.isoformat()

        # Tags
        tags = raw_entry.get("tags", [])
        if not isinstance(tags, list):
            tags = []

        tags += [
            f"date:{dt_local.strftime('%Y-%m-%d')}",
            f"date_friendly:{dt_local.strftime('%B %d, %Y')}",
            f"type:{kind}"
        ]

        # Build payload
        memory = {
            "text": text,
            "tags": list(set(tags)),
            "kind": kind,
            "mood": raw_entry.get("mood"),
            "people": raw_entry.get("people", []),
            "activities": raw_entry.get("activities", []),
            "keywords": raw_entry.get("keywords", []),
            "meta": {
                "datetime_iso": date_iso,
                "timezone": TIMEZONE,
                "version": VERSION
            }
        }

        return memory, warning
    except Exception as e:
        return None, f"normalize_error:{e}"

def migrate_file(filename):
    skipped = []
    migrated = 0

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            try:
                raw_entry = json.loads(line)
            except json.JSONDecodeError:
                skipped.append({"line": line, "reason": "json_decode_error"})
                continue

            memory, warning = normalize_entry(raw_entry)
            if not memory:
                skipped.append({"entry": raw_entry, "reason": warning})
                continue

            # Send to API
            try:
                res = requests.post(API_URL, json=memory)
                if res.status_code != 200:
                    skipped.append({"entry": raw_entry, "reason": f"api_error:{res.text}"})
                else:
                    migrated += 1
                    if warning:
                        skipped.append({"entry": raw_entry, "reason": warning})
            except Exception as e:
                skipped.append({"entry": raw_entry, "reason": f"request_error:{e}"})

    # Save exceptions log
    if skipped:
        with open("exceptions.log", "w", encoding="utf-8") as log:
            for s in skipped:
                log.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"✅ Migrated: {migrated}")
    print(f"⚠️ Skipped/Warnings: {len(skipped)} (see exceptions.log if present)")

if __name__ == "__main__":
    migrate_file("journal_with_tags_and_categories.jsonl")
