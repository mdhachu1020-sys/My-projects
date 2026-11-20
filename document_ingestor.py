import os
from document_loader import DocumentLoader
from text_chunker import TextChunker

class DocumentIngestor:
    def __init__(self, config, vector_store, embedding_model):
        self.config = config
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.loader = DocumentLoader()
        self.chunker = TextChunker()

    def ingest_document(self, file_path: str) -> int:
        print(f"📂 Ingesting: {file_path}")
        raw = self.loader.load(file_path)
        if not raw:
            raise ValueError(f"No text extracted from {file_path}")
        chunks = self.chunker.chunk(raw, self.config.chunk_size, self.config.chunk_overlap)
        if not chunks:
            raise ValueError(f"No chunks from {file_path}")
        print(f"   → {len(chunks)} chunks. Embedding...")
        embeddings = self.embedding_model.embed(chunks)
        fname = os.path.basename(file_path)
        metadata = [{"source_file": fname, "chunk_index": i} for i in range(len(chunks))]
        self.vector_store.add(chunks, embeddings, metadata)
        self.vector_store.save(self.config.vector_store_path)
        print(f"✅ Indexed {len(chunks)} chunks from '{fname}'")
        return len(chunks)

    def ingest_directory(self, directory: str):
        supported = set(self.loader.supported_formats())
        found = False
        for fname in os.listdir(directory):
            if os.path.splitext(fname)[1].lower() in supported:
                found = True
                try:
                    self.ingest_document(os.path.join(directory, fname))
                except Exception as e:
                    print(f"⚠️  Skipped '{fname}': {e}")
        if not found:
            print(f"ℹ️  No supported documents found in '{directory}'.")
