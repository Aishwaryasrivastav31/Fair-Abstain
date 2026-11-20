"""
Build a lightweight FAISS index from documents.
"""

import os
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from utils import load_documents, chunk_text


def build_index(
    doc_dir: str = "data/documents",
    index_path: str = "data/faiss_index.bin",
    meta_path: str = "data/chunks_meta.pkl",
    model_name: str = "BAAI/bge-small-en-v1.5",
    chunk_size: int = 400,
    overlap: int = 50,
):
    print("Loading documents...")
    docs = load_documents(doc_dir)
    if not docs:
        print(f"No documents found in {doc_dir}. Creating dummy documents for demo.")
        os.makedirs(doc_dir, exist_ok=True)
        # Create minimal dummy data so the pipeline can run
        dummy = [
            ("india_history.txt", "Jawaharlal Nehru was the first Prime Minister of India. Mahatma Gandhi led the independence movement. The capital of India is New Delhi."),
            ("world_history.txt", "George Washington was the first President of the United States. The capital of France is Paris. Albert Einstein developed the theory of relativity."),
        ]
        for fname, text in dummy:
            with open(os.path.join(doc_dir, fname), "w") as f:
                f.write(text)
        docs = load_documents(doc_dir)

    print(f"Loaded {len(docs)} documents. Chunking...")
    chunks = []
    meta = []
    for doc in docs:
        doc_chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, ch in enumerate(doc_chunks):
            chunks.append(ch)
            meta.append({"doc_id": doc["id"], "chunk_id": i, "text": ch})

    print(f"Total chunks: {len(chunks)}")
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalisation)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    print(f"Saving index to {index_path}")
    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)

    print("Index built successfully.")
    return index, meta


if __name__ == "__main__":
    build_index()