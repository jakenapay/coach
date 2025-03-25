import chromadb

# Initialize ChromaDB in persistent mode (stores data locally)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or load a collection (similar to a table in SQL)
collection = chroma_client.get_or_create_collection(name="fitness_knowledge")

print("✅ ChromaDB initialized successfully!")