"""
AI Voice Assistant with RAG
Usage:
  python main.py            -> voice mode (microphone)
  python main.py --text     -> text mode (keyboard, no mic needed)
  python main.py --ingest   -> ingest docs only, then exit
"""
import os, sys, argparse

def get_api_key():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    if val and "your_" not in val:
                        return "gemini", val
                if line.startswith("OPENAI_API_KEY="):
                    val = line.split("=", 1)[1].strip()
                    if val and "your_" not in val:
                        return "openai", val
    k = os.getenv("GEMINI_API_KEY", "")
    if k:
        return "gemini", k
    k = os.getenv("OPENAI_API_KEY", "")
    if k:
        return "openai", k
    return None, None


def ensure_sample_docs(docs_dir: str):
    """Create a sample banking notes file if docs folder is empty."""
    os.makedirs(docs_dir, exist_ok=True)
    sample = os.path.join(docs_dir, "banking_notes.txt")
    if not os.listdir(docs_dir):
        with open(sample, "w", encoding="utf-8") as f:
            f.write("""BANKING NOTES - KEY CONCEPTS

Cash Reserve Ratio (CRR)
CRR stands for Cash Reserve Ratio. It is the percentage of a bank's total deposits
that it is required to maintain as liquid cash with the Reserve Bank of India (RBI).
Banks cannot use this money for lending or investment. The current CRR is 4% of NDTL.

Statutory Liquidity Ratio (SLR)
SLR stands for Statutory Liquidity Ratio. It is the minimum percentage of deposits
that a bank must maintain in liquid assets (gold, cash, government securities).
The current SLR in India is 18% of NDTL.

Repo Rate
The Repo Rate is the rate at which the RBI lends money to commercial banks.
A lower repo rate encourages banks to borrow and lend more, stimulating growth.

Reverse Repo Rate
The Reverse Repo Rate is the rate at which the RBI borrows money from commercial banks.
It is always lower than the repo rate and helps absorb excess liquidity.

RTGS (Real-Time Gross Settlement)
RTGS is a funds transfer system where money moves from one bank to another in real-time.
Minimum amount: Rs 2 lakh. Available 24x7x365.

NEFT (National Electronic Funds Transfer)
NEFT is a nation-wide payment system for one-to-one fund transfers, settled in
half-hourly batches. No minimum or maximum amount limit. Available 24x7.

IMPS (Immediate Payment Service)
IMPS offers instant interbank electronic fund transfer via mobile/internet banking.
Available 24x7 including holidays. Limit: up to Rs 5 lakh per transaction.

Non-Performing Asset (NPA)
An NPA is a loan for which principal or interest payment is overdue for 90+ days.
NPAs are classified as Substandard, Doubtful, or Loss assets.

Basel Norms
Basel Norms are banking regulations to ensure banks hold enough capital.
Basel III requires a minimum Capital Adequacy Ratio (CAR) of 9% for Indian banks.

Priority Sector Lending (PSL)
PSL requires banks to lend 40% of their ANBC to priority sectors like Agriculture,
MSMEs, Education, Housing, and Social Infrastructure.

KYC (Know Your Customer)
KYC is the process banks use to verify client identity and assess risk.
It prevents money laundering, terrorist financing, and financial crimes.

SWIFT
SWIFT is a global messaging network banks use to send payment instructions securely.
It does not transfer funds itself but facilitates payment orders between banks.
""")
        print(f"📝 Created sample document: {sample}")


def main():
    parser = argparse.ArgumentParser(description="AI Voice Assistant with RAG")
    parser.add_argument("--text",   action="store_true", help="Text input mode (no mic)")
    parser.add_argument("--ingest", action="store_true", help="Ingest docs and exit")
    args = parser.parse_args()

    # ── API key ──────────────────────────────────────────────────────────────
    provider, api_key = get_api_key()
    if not api_key:
        print("\n❌  No API key found.")
        print("    Open the .env file and set:")
        print("      GEMINI_API_KEY=your_actual_key_here\n")
        sys.exit(1)
    print(f"✅  Using {provider.upper()} API")

    # ── Config ───────────────────────────────────────────────────────────────
    from config import AssistantConfig
    config = AssistantConfig(
        llm_provider=provider,
        api_key=api_key,
        vector_store_path="./data/vector_store",
        docs_dir="./docs",
    )

    # ── Ensure docs exist ────────────────────────────────────────────────────
    ensure_sample_docs(config.docs_dir)

    # ── Components ───────────────────────────────────────────────────────────
    from embedding_model import EmbeddingModel
    from vector_store import VectorStore
    from llm_client import LLMClient
    from rag_engine import RAGEngine
    from tts_engine import TTSEngine
    from document_ingestor import DocumentIngestor
    from exceptions import VectorStoreCorruptionError

    # Pass api_key so embedding model configures genai before LLM client
    emb = EmbeddingModel(api_key=api_key)

    vs = VectorStore()
    try:
        vs.load(config.vector_store_path)
    except VectorStoreCorruptionError:
        print("ℹ️   Starting with empty vector store.")

    llm      = LLMClient(config.llm_provider, config.api_key)
    rag      = RAGEngine(config, vs, emb, llm)
    tts      = TTSEngine(config.tts_engine)
    ingestor = DocumentIngestor(config, vs, emb)

    # ── Ingest-only mode ─────────────────────────────────────────────────────
    if args.ingest:
        ingestor.ingest_directory(config.docs_dir)
        print("✅  Ingestion complete.")
        return

    # ── Auto-ingest if vector store is empty ─────────────────────────────────
    if vs.is_empty():
        print(f"\n📚  Vector store empty — ingesting docs from '{config.docs_dir}' ...")
        ingestor.ingest_directory(config.docs_dir)

    # ── Run assistant ─────────────────────────────────────────────────────────
    if args.text:
        run_text(rag, tts)
    else:
        run_voice(config, rag, tts)


def run_text(rag, tts):
    from exceptions import LLMError
    print("\n" + "="*55)
    print("  🤖  AI Voice Assistant  (text mode)")
    print("  Ask anything — from your docs OR general knowledge.")
    print("  Type 'quit' or Ctrl+C to exit.")
    print("="*55 + "\n")

    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                print("👋  Goodbye!")
                break

            print("⏳  Thinking...")
            result = rag.query(q)

            print(f"\n🤖  {result.answer}")
            if result.source_chunks:
                srcs = ", ".join(sorted({c.source_file for c in result.source_chunks}))
                print(f"    📎  Source: {srcs}")
            else:
                print(f"    🌐  (general knowledge)")
            print(f"    ⏱   {result.latency_ms:.0f} ms\n")

            tts.speak(result.answer)

        except LLMError as e:
            print(f"❌  LLM error: {e}\n")
        except KeyboardInterrupt:
            print("\n👋  Goodbye!")
            break


def run_voice(config, rag, tts):
    from speech_recognizer import SpeechRecognizer
    from exceptions import SpeechRecognitionError, LLMError, MicrophoneNotFoundError

    try:
        mic = SpeechRecognizer()
    except MicrophoneNotFoundError as e:
        print(f"❌  {e}")
        print("    Tip: run with --text flag to use keyboard instead.")
        sys.exit(1)

    print("\n" + "="*55)
    print("  🤖  AI Voice Assistant  (voice mode)")
    print("  Speak your question. Ctrl+C to exit.")
    print("="*55 + "\n")

    while True:
        try:
            q = mic.listen(timeout=config.stt_timeout)
            if not q:
                continue
            print(f"\n🗣️   You: \"{q}\"")
            print("⏳  Thinking...")

            result = rag.query(q)

            print(f"\n🤖  {result.answer}")
            if result.source_chunks:
                srcs = ", ".join(sorted({c.source_file for c in result.source_chunks}))
                print(f"    📎  Source: {srcs}")
            else:
                print(f"    🌐  (general knowledge)")
            print(f"    ⏱   {result.latency_ms:.0f} ms\n")

            tts.speak(result.answer)

        except SpeechRecognitionError as e:
            print(f"⚠️   {e} — please try again.\n")
        except LLMError as e:
            print(f"❌  {e}\n")
            tts.speak("Sorry, I could not reach the AI service right now.")
        except KeyboardInterrupt:
            print("\n👋  Goodbye!")
            break


if __name__ == "__main__":
    main()
