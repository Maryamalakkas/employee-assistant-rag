import chromadb
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


if __name__ == "__main__":
    question = "Who works in the HR department?"
    matches = search(question)

    print(f"Question: {question}\n")
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match}\n")