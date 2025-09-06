import os
from openai import OpenAI
import pinecone

print("🚀 Running query_test.py...")

# === Initialize OpenAI ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Initialize Pinecone ===
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("dayonememories")

# === Your test query ===
query = "Can you tell me about some stressful moments in June?"

# Create embedding for the query
embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# Query Pinecone
results = index.query(
    vector=embedding,
    top_k=5,  # return top 5 most relevant entries
    include_metadata=True
)

# Print results
print(f"\n🔎 Query: {query}\n")
print("📌 Top Matches:")
for match in results.matches:
    text = match.metadata.get("text", "No text found")
    tags = match.metadata.get("tags", [])
    category = match.metadata.get("category", "uncategorized")
    print(f"- Score: {match.score:.3f}")
    print(f"  Tags: {tags}")
    print(f"  Category: {category}")
    print(f"  Text: {text[:200]}...\n")  # print first 200 chars
