import json
import chromadb
import ollama
import os

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="fitness_knowledge")

# Path to the JSON folder
json_folder = os.path.join(os.getcwd(), "json")  # <-- Ensure it points to the json folder

# Insert data from each JSON file
for file in os.listdir(json_folder):
    if file.endswith(".json"):
        file_path = os.path.join(json_folder, file)
        with open(file_path, "r") as f:
            fitness_data = json.load(f)

        # Insert data into ChromaDB
        for idx, text in enumerate(fitness_data):
            embedding = ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

            collection.add(
                ids=[f"{file}_doc_{idx + 1}"],
                documents=[text],
                embeddings=[embedding],
                metadatas=[{"source": file}]
            )
        
        print(f"✅ {file} added to ChromaDB!")

print("🔥 All fitness knowledge inserted into ChromaDB!")
