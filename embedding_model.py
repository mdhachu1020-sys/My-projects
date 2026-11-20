import numpy as np

class EmbeddingModel:
    """Uses Google Gemini embedding API — no local model download needed."""

    def __init__(self, model_name: str = "gemini-embedding-001", api_key: str = ""):
        print("📦 Initialising Google Gemini embedding API ...")
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self._model = "gemini-embedding-001"
        print("✅ Embedding model ready.")

    def embed(self, texts: list) -> list:
        from google.genai import types
        embeddings = []
        for text in texts:
            resp = self._client.models.embed_content(
                model=self._model,
                contents=text,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            embeddings.append(self._normalize(resp.embeddings[0].values))
        return embeddings

    def embed_query(self, query: str) -> list:
        from google.genai import types
        resp = self._client.models.embed_content(
            model=self._model,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return self._normalize(resp.embeddings[0].values)

    def _normalize(self, vec) -> list:
        arr = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()
