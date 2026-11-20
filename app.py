"""
Flask backend for AI Voice Assistant with RAG
"""
import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# ── Global components (loaded once at startup) ───────────────────────────────
rag_engine   = None
tts_engine   = None
ingestor     = None
vector_store = None
config       = None


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
    if k: return "gemini", k
    k = os.getenv("OPENAI_API_KEY", "")
    if k: return "openai", k
    return None, None


def ensure_sample_docs(docs_dir):
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
RTGS is a funds transfer system where money moves in real-time. Minimum: Rs 2 lakh. Available 24x7.

NEFT (National Electronic Funds Transfer)
NEFT is a nation-wide payment system settled in half-hourly batches. No min/max limit. Available 24x7.

IMPS (Immediate Payment Service)
IMPS offers instant interbank fund transfer via mobile/internet. Available 24x7. Limit: Rs 5 lakh.

Non-Performing Asset (NPA)
An NPA is a loan where principal or interest payment is overdue for 90+ days.
NPAs are classified as Substandard, Doubtful, or Loss assets.

Basel Norms
Basel Norms ensure banks hold enough capital. Basel III requires 9% CAR for Indian banks.

Priority Sector Lending (PSL)
PSL requires banks to lend 40% of ANBC to Agriculture, MSMEs, Education, Housing.

KYC (Know Your Customer)
KYC is the process banks use to verify client identity. Prevents money laundering.

SWIFT
SWIFT is a global messaging network for secure payment instructions between banks.
""")


def init_components():
    global rag_engine, tts_engine, ingestor, vector_store, config

    provider, api_key = get_api_key()
    if not api_key:
        print("❌ No API key found in .env")
        sys.exit(1)

    from config import AssistantConfig
    from embedding_model import EmbeddingModel
    from vector_store import VectorStore
    from llm_client import LLMClient
    from rag_engine import RAGEngine
    from tts_engine import TTSEngine
    from document_ingestor import DocumentIngestor
    from exceptions import VectorStoreCorruptionError

    config = AssistantConfig(
        llm_provider=provider,
        api_key=api_key,
        vector_store_path="./data/vector_store",
        docs_dir="./docs",
    )

    ensure_sample_docs(config.docs_dir)

    emb = EmbeddingModel(api_key=api_key)
    vector_store = VectorStore()

    try:
        vector_store.load(config.vector_store_path)
    except VectorStoreCorruptionError:
        print("ℹ️  Fresh vector store.")

    llm          = LLMClient(config.llm_provider, config.api_key)
    rag_engine   = RAGEngine(config, vector_store, emb, llm)
    tts_engine   = TTSEngine(config.tts_engine)
    ingestor     = DocumentIngestor(config, vector_store, emb)

    if vector_store.is_empty():
        print("📚 Auto-ingesting docs...")
        ingestor.ingest_directory(config.docs_dir)

    print("✅ All components ready. Server starting...")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("message", "").strip()
    if not question:
        return jsonify({"error": "Empty message"}), 400

    try:
        result = rag_engine.query(question)
        sources = list({c.source_file for c in result.source_chunks})
        return jsonify({
            "answer": result.answer,
            "sources": sources,
            "used_fallback": result.used_fallback,
            "latency_ms": round(result.latency_ms),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/speak", methods=["POST"])
def speak():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text"}), 400
    try:
        tts_engine.speak(text)
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "No filename"}), 400

    allowed = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format. Use: {', '.join(allowed)}"}), 400

    save_path = os.path.join(config.docs_dir, f.filename)
    f.save(save_path)

    try:
        count = ingestor.ingest_document(save_path)
        return jsonify({"status": "ok", "chunks": count, "filename": f.filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/docs", methods=["GET"])
def list_docs():
    docs_dir = config.docs_dir
    files = []
    if os.path.isdir(docs_dir):
        for fname in os.listdir(docs_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext in {".pdf", ".docx", ".txt"}:
                size = os.path.getsize(os.path.join(docs_dir, fname))
                files.append({"name": fname, "size": size})
    return jsonify({"docs": files})


if __name__ == "__main__":
    init_components()
    app.run(debug=False, port=5000)
