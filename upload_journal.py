import json
import requests

# === CONFIG ===
API_URL = "https://core-memory-api.onrender.com/storeMemory"
INPUT_FILE = "journal_fixed.json"

# === FUNCTIONS ===

def upload_entries(entries):
    uploaded = []
    for e in entries:
        entry = {
            "text": e.get("text", ""),
            "kind": e.get("kind", "note"),
            "title": e.get("title", ""),
            "tags": e.get("tags", []),
            "mood": e.get("mood", ""),
            "people": e.get("people", []),
            "activities": e.get("activities", []),
            "keywords": e.get("keywords", []),
            "meta": {
                "datetime_iso": e.get("meta", {}).get("datetime_iso", ""),
                "timezone": e.get("meta", {}).get("timezone", ""),
                "version": e.get("meta", {}).get("version", "v3.7a")
            }
        }
        uploaded.append(entry)
    return uploaded

def main():
    # Load entries
    with open(INPUT_FILE, "r") as f:
        entries = json.load(f)

    # Ensure all entries have a meta field
    for e in entries:
        if "meta" not in e:
            e["meta"] = {
                "datetime_iso": "",
                "timezone": "",
                "version": "v3.7a"
            }

    # Prepare upload payload
    payload = upload_entries(entries)

    # Upload to API
    for entry in payload:
        try:
            response = requests.post(API_URL, json=entry)
            if response.status_code == 200:
                print(f"✅ Uploaded: {entry.get('title','(no title)')}")
            else:
                print(f"❌ Failed: {entry.get('title','(no title)')} | {response.text}")
        except Exception as ex:
            print(f"⚠️ Error uploading entry: {entry.get('title','(no title)')} | {ex}")

if __name__ == "__main__":
    main()
