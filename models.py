from dataclasses import dataclass, field

@dataclass
class DocumentChunk:
    text: str
    source_file: str
    page_number: int | None
    chunk_index: int
    similarity_score: float = 0.0

@dataclass
class QueryResult:
    question: str
    answer: str
    source_chunks: list = field(default_factory=list)
    used_fallback: bool = False
    latency_ms: float = 0.0
