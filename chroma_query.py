import chromadb
import ollama

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="fitness_knowledge")

# User question
query = input("\n🔍 Enter your fitness question: ")

# Generate embedding for the query
query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]

# Query ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3  # Retrieve top 3 matches
)

# Display the results
print("\n🔍 Top Matches:")
for i, doc in enumerate(results["documents"][0]):
    print(f"{i + 1}. {doc}")
