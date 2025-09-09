import json

INPUT_FILE = "core_memory_api/journal_with_tags_and_categories.jsonl"
OUTPUT_FILE = "core_memory_api/journal_fixed.jsonl"

with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            entry = json.loads(line)

            # Ensure every entry has a kind
            if "kind" not in entry:
                entry["kind"] = "journal"

            # Fix meta: remove if it's empty {}
            if "meta" in entry:
                if not entry["meta"]:  # empty dict {}
                    del entry["meta"]

            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

        except json.JSONDecodeError:
            print("⚠️ Skipping invalid JSON line:", line.strip())

print(f"✅ Migration complete. Cleaned file saved as {OUTPUT_FILE}")
