import chromadb
import ollama
import random
import time

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="fitness_knowledge")

def retrieve_knowledge(query, k=50, max_results=10):
    """Retrieve diverse and comprehensive knowledge from ChromaDB."""
    try:
        query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]

        # Retrieve more documents for broader coverage
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        # Remove duplicates and shuffle
        retrieved_docs = list(set(results['documents'][0]))
        random.shuffle(retrieved_docs)

        # Return the top unique results
        num_results = min(len(retrieved_docs), max_results)
        return retrieved_docs[:num_results] if retrieved_docs else ["No relevant knowledge found."]

    except Exception as e:
        print(f"⚠️ Retrieval Error: {e}")
        return ["No knowledge retrieved due to an error."]

def main():
    print("\n🔥 Welcome to your Enhanced RAG-Powered Fitness Coach! 🔥")

    while True:
        query = input("\n🔍 Enter your fitness question (or type 'exit' to quit): ")

        if query.lower() == "exit":
            print("👋 Exiting. Stay fit, boss! 💪")
            break

        start_time = time.time()

        # Retrieve diverse knowledge
        retrieved_knowledge = retrieve_knowledge(query)

        # Format retrieved knowledge for the LLM prompt
        knowledge_str = "\n".join([f"- {doc}" for doc in retrieved_knowledge])

        # 🔥 Fine-tuned prompt with structured sections and expert-level instructions
        prompt = f"""
You are a highly knowledgeable and experienced fitness coach assistant. 
Your goal is to provide expert, structured, and detailed fitness advice by combining the retrieved knowledge and your own expertise into a **single, seamless, and cohesive answer**.

### Context:
- You are a fitness coach with expertise in calisthenics, strength training, mobility, and nutrition.
- You are advising athletes, gym-goers, and fitness enthusiasts.
- Use the retrieved knowledge as background information.
- Seamlessly integrate it into your response without explicitly listing it.
- If the knowledge is insufficient, add your own expert insights.

### Retrieved Knowledge:
{knowledge_str}

### User Question:
{query}

### Instructions:
- **Provide a professional, structured, and expert-level answer**.
- **Include practical examples**.
- **Use markdown formatting** for readability.
- **Add key takeaways** at the end with bullet points.
"""

        # Query the LLM with the refined prompt
        try:
            response = ollama.chat(model="deepseek-r1:7b", messages=[
                {"role": "user", "content": prompt}
            ])

            # Display retrieved knowledge first
            print("\n🔍 **Retrieved Knowledge:**")
            for idx, doc in enumerate(retrieved_knowledge, 1):
                print(f"{idx}. {doc}")

            # Display the formatted response
            print("\n💡 **AI Coach Response:**\n")
            print(response['message']['content'])

        except Exception as e:
            print(f"⚠️ LLM Error: {e}")

        end_time = time.time()
        print(f"\n⏱️ **Response Time:** {round(end_time - start_time, 2)} seconds")

if __name__ == "__main__":
    main()
