import os
from dataclasses import dataclass, field

@dataclass
class AssistantConfig:
    llm_provider: str = "gemini"
    api_key: str = ""
    embedding_model: str = "models/text-embedding-004"
    vector_store_type: str = "faiss"
    vector_store_path: str = "./data/vector_store"
    chunk_size: int = 500
    chunk_overlap: int = 50
    retrieval_top_k: int = 5
    retrieval_threshold: float = 0.4
    tts_engine: str = "pyttsx3"
    stt_timeout: float = 5.0
    docs_dir: str = "./docs"
