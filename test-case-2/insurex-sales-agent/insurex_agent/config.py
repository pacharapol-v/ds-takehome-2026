"""Central configuration (12-factor style: everything overridable by env / .env)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM -----------------------------------------------------------------
    # "offline" = no LLM; deterministic rule-based brain (tests / degraded mode)
    llm_provider: Literal["openai", "google_genai", "anthropic", "ollama", "offline"] = "openai"
    llm_model: str = "gpt-5-mini"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2
    llm_reasoning_effort: str | None = "low"   # only used by reasoning models (gpt-5*, o*)
    verify_with_llm: bool = True               # LLM fact-check on top of the deterministic number check

    # --- Embeddings ------------------------------------------------------------
    # "local" = offline TF-IDF + SVD (LSA) embedder fitted on the corpus.
    embedding_provider: Literal["openai", "google_genai", "huggingface", "ollama", "local"] = "openai"
    embedding_model: str = "text-embedding-3-small"

    # --- Paths -------------------------------------------------------------------
    pdf_dir: Path = ROOT / "data" / "pdfs"
    processed_dir: Path = ROOT / "data" / "processed"
    index_dir: Path = ROOT / "data" / "index"
    state_dir: Path = ROOT / "data" / "state"

    # --- Retrieval ---------------------------------------------------------------
    retrieval_k_dense: int = 12
    retrieval_k_bm25: int = 12
    rrf_k: int = 60
    section_boosts: bool = False           # off: the ablation showed they slightly hurt Hit@1 (eval/results)
    max_context_parents: int = 4
    max_retrieval_attempts: int = 2        # retrieve -> grade -> rewrite -> retrieve ...
    max_generation_attempts: int = 2       # generate -> verify -> regenerate ...

    # --- Memory ------------------------------------------------------------------
    history_window: int = 8                # messages sent verbatim to the LLM
    summarize_after: int = 16              # older messages are folded into a summary

    # --- Leads / MCP --------------------------------------------------------------
    leads_backend: Literal["mcp", "direct"] = "mcp"

    # --- Misc -------------------------------------------------------------------------
    contact_center: str = "InsureX Contact Center 1314"
    log_level: str = "INFO"

    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def leads_db_path(self) -> Path:
        return self.state_dir / "leads.sqlite"

    @property
    def checkpoint_db_path(self) -> Path:
        return self.state_dir / "checkpoints.sqlite"

    @property
    def sessions_db_path(self) -> Path:
        return self.state_dir / "sessions.sqlite"

    @property
    def collection_name(self) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in f"{self.embedding_provider}_{self.embedding_model}")
        return f"insurex_kb_{safe}"[:60]

    def ensure_dirs(self) -> None:
        for d in (self.processed_dir, self.index_dir, self.state_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)   # provider SDKs read OPENAI_API_KEY etc. from os.environ
    s = Settings()
    s.ensure_dirs()
    return s


def reset_settings_cache() -> None:
    get_settings.cache_clear()

