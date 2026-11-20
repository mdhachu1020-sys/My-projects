import time
from models import QueryResult

class RAGEngine:
    def __init__(self, config, vector_store, embedding_model, llm_client):
        self.config = config
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm_client = llm_client

    def query(self, user_question: str) -> QueryResult:
        start = time.time()
        q = user_question.strip()

        # 1. Embed the question
        query_emb = self.embedding_model.embed_query(q)

        # 2. Search vector store
        chunks = self.vector_store.search(query_emb, top_k=self.config.retrieval_top_k)
        relevant = [c for c in chunks if c.similarity_score >= self.config.retrieval_threshold]

        # 3. Build prompt
        if relevant:
            context = "\n\n".join(c.text for c in relevant)
            prompt = (
                "You are a helpful AI assistant. Use the context below to answer the question.\n"
                "If the answer is not in the context, answer from your general knowledge anyway.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {q}\n"
                "Answer:"
            )
            used_fallback = False
            print(f"   📄  {len(relevant)} relevant chunk(s) — document-grounded answer.")
        else:
            # NO relevant docs — answer from general knowledge
            prompt = (
                "You are a helpful AI assistant. Answer the following question clearly and concisely "
                "using your general knowledge.\n\n"
                f"Question: {q}\n"
                "Answer:"
            )
            used_fallback = True
            print("   🌐  No relevant chunks — answering from general knowledge.")

        # 4. Generate answer
        answer = self.llm_client.generate(prompt)
        latency = (time.time() - start) * 1000

        return QueryResult(
            question=q,
            answer=answer,
            source_chunks=relevant,
            used_fallback=used_fallback,
            latency_ms=latency,
        )
