import json
import chromadb
from numpy.ma import ids
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed/employee_chunks.json"
DB_PATH = "data/processed/chroma_db"


def load_chunks():
    with open(CHUNKS_PATH) as f:
        return json.load(f)


def build_vector_store():
    chunks = load_chunks()

    # this model turns text into vectors - small and fast, runs fine on cpu
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # chromadb will persist to disk here, so we don't have to re-embed every time
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name="employees")

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]

    print(f"Embedding {len(texts)} chunks, this might take a minute...")
    embeddings = model.encode(texts, show_progress_bar=True)
    # chromadb has a max batch size, so we add the data in chunks instead of all at once
    batch_size = 1000
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size].tolist(),
            documents=texts[i:i + batch_size]
        )
        print(f"Added batch {i // batch_size + 1}, total so far: {min(i + batch_size, len(ids))}")

    print(f"Done. Vector store now has {collection.count()} entries.")


if __name__ == "__main__":
    build_vector_store()



'''
error size limit exceeded when adding embeddings to chromadb, so we add them in batches
    chromadb wants embeddings as plain lists, not numpy arrays
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts
    )
'''