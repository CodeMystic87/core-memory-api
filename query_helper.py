import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def universal_query(index, date=None, keyword=None, tag=None, semantic=None, top_k=3):
    # Step 1: Index stats
    print("📊 Step 1: Index stats")
    stats = index.describe_index_stats()
    print(stats)

    # Step 2: Peek at metadata keys
    print("\n🧾 Step 2: Sample metadata from one entry")
    res_any = index.query(
        vector=[0]*1536,
        top_k=1,
        include_metadata=True
    )
    if res_any["matches"]:
        print("Available keys:", res_any["matches"][0]["metadata"].keys())
    else:
        print("⚠️ No matches found at all in index.")
        return

    # Step 3: Run query
    if date:
        print(f"\n🔍 Step 3: Querying for date = {date}")
        res = index.query(
            vector=[0]*1536,
            top_k=top_k,
            include_metadata=True,
            filter={"date": {"$eq": date}}
        )
    elif keyword:
        print(f"\n🔍 Step 3: Querying for keyword = '{keyword}' in text")
        res = index.query(
            vector=[0]*1536,
            top_k=top_k,
            include_metadata=True,
            filter={"text": {"$contains": keyword}}
        )
    elif tag:
        print(f"\n🔍 Step 3: Querying for tag = '{tag}' in tags")
        res = index.query(
            vector=[0]*1536,
            top_k=top_k,
            include_metadata=True,
            filter={"tags": {"$contains": tag}}
        )
    elif semantic:
        print(f"\n🤖 Step 3: Semantic search for '{semantic}'")
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=semantic
        ).data[0].embedding
        res = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
    else:
        print("\n⚠️ No date, keyword, tag, or semantic provided.")
        return

    # Step 4: Print results
    if not res["matches"]:
        print("⚠️ No results found.")
    else:
        for match in res["matches"]:
            print("📌 Title:", match["metadata"].get("title"))
            print("📝 Text:", match["metadata"].get("text")[:200])
            print("Date:", match["metadata"].get("date"))
            print("Tags:", match["metadata"].get("tags"))
            print("-" * 50)
