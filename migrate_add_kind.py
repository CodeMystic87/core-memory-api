import json

# Input and output paths
INPUT_FILE = "core_memory_api/journal_with_tags_and_categories.jsonl"
OUTPUT_FILE = "core_memory_api/journal_fixed.jsonl"

def migrate_add_kind():
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        for line in infile:
            entry = json.loads(line)

            # If kind is missing, set it to "journal"
            if "kind" not in entry:
                entry["kind"] = "journal"

            outfile.write(json.dumps(entry) + "\n")

    print(f"✅ Migration complete. Fixed file saved as {OUTPUT_FILE}")

if __name__ == "__main__":
    migrate_add_kind()
