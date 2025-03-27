import chromadb
import ollama
import random

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="fitness_knowledge")

# User question
query = input("\n🔍 Enter your fitness question: ")

# Generate embedding for the query
query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]

# Query ChromaDB with a large batch size
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=20  # Retrieve a larger batch
)

# Extract and randomize the results
retrieved_docs = list(set(results["documents"][0]))  # Remove duplicates
random.shuffle(retrieved_docs)

# Display 5 unique results (or fewer if limited)
num_to_display = min(len(retrieved_docs), 5)

if num_to_display > 0:
    print("\n🔍 Top Matches:")
    for i, doc in enumerate(retrieved_docs[:num_to_display]):
        print(f"{i + 1}. {doc}")
else:
    print("\n⚠️ No results found. Try a different query.")
