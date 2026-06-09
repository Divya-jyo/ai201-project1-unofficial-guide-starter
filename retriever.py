"""Embed document chunks and retrieve the most relevant ones for a query.

Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings and an in-memory
ChromaDB collection for storage and similarity search.
"""

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import get_chunks

MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "professor_reviews"

# Loaded lazily so importing this module is cheap.
_model = None
_collection = None


def get_model():
    """Return the shared embedding model, loading it on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts):
    """Embed a list of texts into plain Python lists (ChromaDB-friendly)."""
    embeddings = get_model().encode(list(texts), show_progress_bar=False)
    return embeddings.tolist()


def build_index(records=None):
    """Embed all chunks and store them in a fresh ChromaDB collection.

    Each chunk is stored with metadata: source filename and chunk index.
    Returns the populated collection.
    """
    global _collection

    if records is None:
        records = get_chunks()
    if not records:
        raise ValueError("No chunks to index. Are the documents populated?")

    client = chromadb.Client()
    # Start from a clean slate so re-runs don't duplicate documents.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    documents = [r["text"] for r in records]
    metadatas = [
        {"source": r["source"], "chunk_index": r["chunk_index"]} for r in records
    ]
    ids = [f"{r['source']}-{r['chunk_index']}" for r in records]

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embed(documents),
        metadatas=metadatas,
    )

    _collection = collection
    return collection


def retrieve(query, top_k=4):
    """Return the top_k most relevant chunks for a query string.

    Builds the index on first call if it hasn't been built yet.
    Returns a list of dicts: text, source, chunk_index, distance.
    """
    global _collection
    if _collection is None:
        build_index()

    results = _collection.query(
        query_embeddings=embed([query]),
        n_results=top_k,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "text": doc,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "distance": dist,
        }
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def main():
    records = get_chunks()
    print(f"Embedding {len(records)} chunks with {MODEL_NAME}...")
    build_index(records)

    query = "What do students say about Professor Abiodun Robert?"
    print(f"\nQuery: {query}\n")

    for rank, result in enumerate(retrieve(query, top_k=4), start=1):
        print(f"[{rank}] {result['source']} (chunk {result['chunk_index']}) "
              f"| distance: {result['distance']:.4f}")
        print(f"    {result['text']}\n")


if __name__ == "__main__":
    main()
