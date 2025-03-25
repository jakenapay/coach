import json
import chromadb
import ollama
import os

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="fitness_knowledge")

# Load data from JSON folder inside the coach project
json_path = os.path.join("json", "fitness_data.json")

# Read the fitness data
with open(json_path, "r") as f:
    fitness_data = json.load(f)

# Insert data into ChromaDB
for idx, text in enumerate(fitness_data):
    # Generate embeddings
    embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

    # Add data to ChromaDB
    collection.add(
        ids=[f"doc_{len(collection.get()['ids']) + 1}"],
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"source": "fitness_tips"}]
    )

print("✅ More fitness data added to ChromaDB!")
