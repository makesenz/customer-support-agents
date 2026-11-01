"""Central settings, read once from the environment (.env)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# --- LLM -------------------------------------------------------------------
OPENAI_API_KEY = _env("OPENAI_API_KEY")
LLM_MODEL = _env("LLM_MODEL", "gpt-4o-mini")
# Optional: any OpenAI-compatible endpoint (Groq, Gemini, Ollama, OpenRouter...)
LLM_BASE_URL = _env("LLM_BASE_URL")
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0.2"))

# --- Paths -----------------------------------------------------------------
KB_DIR = ROOT / "knowledge_base"
INTERNAL_KB_DIR = KB_DIR / "internal"
GRAPH_SEED = ROOT / "knowledge_graph" / "graph_seed.yaml"
DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "data")))
INDEX_DIR = DATA_DIR / "index"
LOG_DIR = Path(_env("LOG_DIR", str(ROOT / "logs")))
TICKET_LOG = LOG_DIR / "tickets.csv"

# --- Retrieval -------------------------------------------------------------
EMBED_MODEL = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = _env("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
MODEL_CACHE_DIR = _env("MODEL_CACHE_DIR", str(DATA_DIR / "models"))
TOP_K_EACH = int(_env("TOP_K_EACH", "8"))  # candidates from BM25 and from FAISS
TOP_K_FINAL = int(_env("TOP_K_FINAL", "4"))  # passages after reranking
RRF_K = 60
# Rerank logit below which a passage is treated as "not relevant".
RERANK_MIN_SCORE = float(_env("RERANK_MIN_SCORE", "-2.0"))

# --- Knowledge graph (Neo4j) ------------------------------------------------
NEO4J_URI = _env("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = _env("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _env("NEO4J_PASSWORD")
GRAPH_ENABLED = _env("GRAPH_ENABLED", "true").lower() == "true"

# --- LangSmith tracing -----------------------------------------------------
# The langsmith SDK reads LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT itself.
LANGSMITH_ENABLED = _env("LANGSMITH_TRACING", "false").lower() == "true" and bool(_env("LANGSMITH_API_KEY"))

# --- Business facts used by guardrails -------------------------------------
BRAND_NAME = "Maa Wellness"
OFFICIAL_WEBSITE = "www.maawellness.in"
SUPPORT_PHONE = "+91 90425 12304"
SUPPORT_EMAIL = "maanaturalsmrperfect@gmail.com"
OFFICIAL_UPI = "maawellness@upi"
OFFICIAL_ACCOUNT_NUMBER = "00000000000000"
OFFICIAL_IFSC = "HDFC0000000"
