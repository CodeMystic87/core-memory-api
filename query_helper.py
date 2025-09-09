def universal_query(index, date=None, keyword=None, tag=None, semantic=None, top_k=3):
    # Step 1: Index stats
    stats = index.describe_index_stats()
    print("📊 Index Stats:", stats)

    # Step 2: Peek at metadata keys
    res_any = index.query(vector=[0]*1536, top_k=1, include_metadata=True)
    if res_any.get("matches"):
        print("\n🧾 Sample metadata keys:", res_any["matches"][0].get("metadata", {}).keys())
    else:
        print("⚠️ No matches found at all in index.")
        return

    # Step 3: Build filter (for date or tag)
    pinecone_filter = {}
    if date:
        pinecone_filter["date"] = {"$eq": date}
    if tag:
        pinecone_filter["tags"] = {"$contains": tag}

    # Step 4: Decide query type
    if keyword:
        print(f"\n🔍 Keyword search = {keyword}")
        # First attempt: direct semantic embedding of keyword
        client = get_openai_client()
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

        # Fallback: if no matches, broaden semantic search
        if not res or "matches" not in res or not res["matches"]:
            print("⚠️ No exact keyword matches, falling back to semantic search...")
            semantic = keyword  # reuse as semantic
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

    elif semantic:
        print(f"\n🤖 Semantic search for '{semantic}'")
        client = get_openai_client()
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

    # Step 5: Print results safely
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
