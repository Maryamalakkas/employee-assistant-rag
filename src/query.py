import chromadb
import ollama
from sentence_transformers import SentenceTransformer

DB_PATH = "data/processed/chroma_db"


def search(question, n_results=3):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name="employees")

    # turn the question into a vector the same way we did for the chunks
    question_embedding = model.encode([question])

    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=n_results
    )

    return results["documents"][0]


def generate_answer(question, context_chunks):
    # join the retrieved employee sentences into one block of context
    context = "\n".join(context_chunks)

    prompt = f"""You are an internal assistant answering questions about employees.
Use ONLY the information below to answer the question. If the answer isn't in the context, say you don't have that information.

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    question = "Tell me about Debra Ponce role and experience"
    matches = search(question)
    answer = generate_answer(question, matches)

    print(f"Question: {question}\n")
    print(f"Answer: {answer}")