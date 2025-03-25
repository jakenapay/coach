import chromadb
import ollama

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="fitness_knowledge")

# User question
query = input("\n🔍 Ask your fitness question: ")

# Retrieve top matches from ChromaDB
query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# Combine retrieved results into context
context = "\n".join(results["documents"][0])

# Generate RAG-powered response
print("\n🤖 Generating RAG-powered answer...")

response = ollama.chat(
    model="deepseek-r1:7b",
    messages=[
        {"role": "system", "content": "You are a fitness coach providing helpful workout and health tips."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
    ]
)

# Display the answer
print("\n🧠 RAG-Powered Answer:")
print(response['message']['content'])
