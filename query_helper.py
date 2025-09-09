import os
from openai import OpenAI

def get_openai_client():
    """Return a reusable OpenAI client"""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def universal_query(index, date=None, keyword=None, tag=None, semantic=None, top_k=3):
    """Universal query helper for Pinecone with literal + semantic support"""

    # 1️⃣ Show index stats
    stats = index.describe_index_stats()
    print("📊 Index Stats:", stats)

    # 2️⃣ Build Pinecone filters
    pinecone_filter = {}
    if date:
        pinecone_filter["date"] = {"$eq": date}
    if tag:
        pinecone_filter["tags"] = {"$contains": tag}

    client = get_openai_client()
    res = None

    # 3️⃣ Handle keyword search
    if keyword:
        print(f"\n🔍 Keyword search = {keyword}")

        # Semantic attempt first
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

        # Literal fallback if no matches
        if not res or "matches" not in res or not res["matches"]:
            print("⚠️ No semantic matches, trying literal text filter...")
            semantic_res = index.query(
                vector=[0]*1536,
                top_k=50,  # pull a larger pool
                include_metadata=True,
                filter=pinecone_filter if pinecone_filter else None
            )

            matches = []
            for match in semantic_res.get("matches", []):
                text = match["metadata"].get("text", "").lower()
                if keyword.lower() in text:
                    matches.append(match)

            res = {"matches": matches[:top_k]}

    # 4️⃣ Handle semantic-only search
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

    # 5️⃣ Handle date/tag-only search
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

    # 6️⃣ Print results safely
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

