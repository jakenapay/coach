import json
import os
import chromadb
import ollama

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="fitness_knowledge")

# Path to the json folder
json_folder = "./json"

# Iterate through all JSON files
for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        json_path = os.path.join(json_folder, filename)
        
        with open(json_path, "r") as f:
            data = json.load(f)

        # Insert each tip into ChromaDB
        for idx, text in enumerate(data):
            embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

            collection.add(
                ids=[f"doc_{len(collection.get()['ids']) + 1}"],
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"source": filename}]
            )

        print(f"✅ {filename} added to ChromaDB!")

print("🔥 All fitness knowledge inserted into ChromaDB!")
