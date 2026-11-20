import os
import pickle
import numpy as np
from models import DocumentChunk
from exceptions import VectorStoreCorruptionError

class VectorStore:
    def __init__(self):
        self.chunks = []
        self.metadata = []
        self._index = None
        self._dim = None

    def add(self, chunks: list, embeddings: list, metadata: list):
        import faiss
        vecs = np.array(embeddings, dtype=np.float32)
        if self._index is None:
            self._dim = vecs.shape[1]
            self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(vecs)
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)

    def search(self, query_embedding: list, top_k: int = 5) -> list:
        if self._index is None or self._index.ntotal == 0:
            return []
        import faiss
        q = np.array([query_embedding], dtype=np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self.metadata[idx]
            results.append(DocumentChunk(
                text=self.chunks[idx],
                source_file=meta.get("source_file", "unknown"),
                page_number=meta.get("page_number"),
                chunk_index=meta.get("chunk_index", int(idx)),
                similarity_score=float(score),
            ))
        results.sort(key=lambda c: c.similarity_score, reverse=True)
        return results

    def save(self, path: str):
        import faiss
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump({"chunks": self.chunks, "metadata": self.metadata, "dim": self._dim}, f)
        print(f"💾 Vector store saved → {path}")

    def load(self, path: str):
        import faiss
        idx_path = os.path.join(path, "index.faiss")
        dat_path = os.path.join(path, "data.pkl")
        if not os.path.exists(idx_path):
            raise VectorStoreCorruptionError(f"No saved index at {path}")
        self._index = faiss.read_index(idx_path)
        with open(dat_path, "rb") as f:
            d = pickle.load(f)
        self.chunks = d["chunks"]
        self.metadata = d["metadata"]
        self._dim = d["dim"]
        print(f"✅ Vector store loaded — {len(self.chunks)} chunks")

    def is_empty(self):
        return self._index is None or self._index.ntotal == 0
