import json

INPUT_FILE = "journal_with_tags_and_categories.jsonl"
OUTPUT_FILE = "journal_fixed.jsonl"

fixed_count = 0
skipped_count = 0

with open(INPUT_FILE, "r", encoding="utf-8") as infile, open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            entry = json.loads(line)

            # Ensure "text" exists and is not empty
            if not entry.get("text") or not entry["text"].strip():
                skipped_count += 1
                continue

            # Add "kind": "journal" if missing
            if "kind" not in entry:
                entry["kind"] = "journal"

            # Write cleaned entry
            outfile.write(json.dumps(entry) + "\n")
            fixed_count += 1

        except Exception as e:
            print(f"⚠️ Skipped bad line: {e}")
            skipped_count += 1

print(f"✅ Migration complete. Fixed: {fixed_count}, Skipped: {skipped_count}")
print(f"➡️ Cleaned file saved as {OUTPUT_FILE}")
