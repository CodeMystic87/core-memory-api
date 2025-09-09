def universal_query(index, date=None, keyword=None, tag=None, semantic=None, top_k=3):
    stats = index.describe_index_stats()
    print("📊 Index Stats:", stats)

    # Build filters
    pinecone_filter = {}
    if date:
        pinecone_filter["date"] = {"$eq": date}
    if tag:
        pinecone_filter["tags"] = {"$contains": tag}

    client = get_openai_client()
    res = None

    if keyword:
        print(f"\n🔍 Keyword search = {keyword}")

        # 1️⃣ Try semantic embedding search first
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=keyword
        ).data[0].embedding

        res = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter if pinecone_filter else None
        )

        # 2️⃣ If no matches, try literal fallback
        if not res or "matches" not in res or not res["matches"]:
            print("⚠️ No semantic matches, trying literal text filter...")
            text_filter = {"text": {"$contains": keyword}}
            if pinecone_filter:
                text_filter.update(pinecone_filter)
            res = index.query(
                vector=[0]*1536,
                top_k=top_k,
                include_metadata=True,
                filter=text_filter
            )

    elif semantic:
        print(f"\n🤖 Semantic search for '{semantic}'")
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=semantic
        ).data[0].embedding
        res = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter if pinecone_filter else None
        )

    elif pinecone_filter:
        print(f"\n🔍 Querying by filter only = {pinecone_filter}")
        res = index.query(
            vector=[0]*1536,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter
        )

    else:
        print("\n⚠️ No date, keyword, tag, or semantic provided.")
        return

    # Print results safely
    if not res or "matches" not in res or not res["matches"]:
        print("⚠️ No results found.")
        return

    for match in res["matches"]:
        meta = match.get("metadata", {})
        print("📌 Title:", meta.get("title"))
        print("📝 Text:", meta.get("text", "")[:200])
        print("Date:", meta.get("date"))
        print("Tags:", meta.get("tags"))
        print("-" * 50)
