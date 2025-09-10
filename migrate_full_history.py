import json
import requests
from datetime import datetime
import pytz

API_URL = "https://core-memory-api.onrender.com/storeMemory"
TIMEZONE = "America/Phoenix"
VERSION = "v3.8"

def detect_kind(text):
    """Guess kind based on leading text markers."""
    lowered = text.strip().lower()
    if lowered.startswith("decision:"):
        return "decision"
    if lowered.startswith("conversation:") or lowered.startswith("dialogue:"):
        return "conversation"
    if lowered.startswith("goal:"):
        return "goal"
    if lowered.startswith("reference:"):
        return "reference"
    return "journal"

def migrate_file(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    success, skipped, errors = 0, 0, 0

    with open("migration_log.txt", "w", encoding="utf-8") as log:
        for idx, line in enumerate(lines, 1):
            try:
                entry = json.loads(line)

                text = entry.get("text", "").strip()
                if not text or len(text) < 20:
                    skipped += 1
                    log.write(f"[SKIPPED] {idx}: too short or empty\n")
                    continue

                # Normalize date
                date_str = entry.get("date")
                if not date_str:
                    skipped += 1
                    log.write(f"[SKIPPED] {idx}: missing date\n")
                    continue

                try:
                    date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except Exception:
                    skipped += 1
                    log.write(f"[SKIPPED] {idx}: invalid date format\n")
                    continue

                local_tz = pytz.timezone(TIMEZONE)
                local_dt = date.astimezone(local_tz)

                # Kind detection
                kind = detect_kind(text)

                # Build payload
                payload = {
                    "text": text,
                    "kind": kind,
                    "tags": [
                        f"date:{local_dt.strftime('%Y-%m-%d')}",
                        f"date_friendly:{local_dt.strftime('%B %d, %Y')}",
                        f"type:{kind}"
                    ],
                    "meta": {
                        "datetime_iso": local_dt.isoformat(),
                        "timezone": TIMEZONE,
                        "version": VERSION
                    }
                }

                # Send to CoreMemory API
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    success += 1
                    log.write(f"[SUCCESS] {idx}: saved ({kind})\n")
                else:
                    errors += 1
                    log.write(f"[ERROR] {idx}: status {response.status_code}\n")

            except Exception as e:
                errors += 1
                log.write(f"[ERROR] {idx}: exception {str(e)}\n")

    print(f"Migration complete. ✅ {success} saved, ⚠️ {skipped} skipped, ❌ {errors} errors, out of {total} entries.")
    print("See migration_log.txt for details.")
