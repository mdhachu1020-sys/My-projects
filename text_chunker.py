class TextChunker:
    def chunk(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list:
        if not text.strip():
            return []
        words = text.split()
        chunks = []
        step = max(chunk_size - overlap, 1)
        start = 0
        while start < len(words):
            chunk = " ".join(words[start:start + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
            start += step
        return chunks
