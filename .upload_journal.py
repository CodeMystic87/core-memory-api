batch_size = 100  # adjust as needed

for i in range(0, len(entries), batch_size):
    batch = entries[i:i + batch_size]

    # Extract texts safely
    texts = [str(entry["text"]) for entry in batch]

    print(f"Processing batch {i // batch_size + 1} with {len(texts)} entries...")

    # Generate embeddings
    embeddings = client.embeddings.create(
        model="text-embedding-3-small",   # or "text-embedding-3-large" for higher quality
        input=texts
    )

    # Prepare vectors for Pinecone
    vectors = [
        {
            "id": f"entry-{i+j}",
            "values": embeddings.data[j].embedding,
            "metadata": {
                "tag": entry.get("tag", ""),
                "category": entry.get("category", ""),
                "text": entry["text"]
            }
        }
        for j, entry in enumerate(batch)
    ]

    # Upload to Pinecone
    index.upsert(vectors)

print("✅ Finished uploading all journal entries to Pinecone!")
