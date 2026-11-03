"""Factories for chat models and embeddings (provider-agnostic)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings

log = logging.getLogger(__name__)


_KEYS = {"openai": "OPENAI_API_KEY", "google_genai": "GOOGLE_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def check_credentials(settings: Settings | None = None, need_llm: bool = True) -> None:
    """Fail fast with an actionable message instead of a provider stack trace."""
    import os

    s = settings or get_settings()
    missing = []
    if need_llm and s.llm_provider in _KEYS and not os.environ.get(_KEYS[s.llm_provider]):
        missing.append(f"{_KEYS[s.llm_provider]} (LLM_PROVIDER={s.llm_provider})")
    if s.embedding_provider in _KEYS and not os.environ.get(_KEYS[s.embedding_provider]):
        missing.append(f"{_KEYS[s.embedding_provider]} (EMBEDDING_PROVIDER={s.embedding_provider})")
    if missing:
        raise SystemExit("Missing API key: " + ", ".join(missing) + ". Put it in .env (see .env.example), "
                         "or run keyless with LLM_PROVIDER=offline EMBEDDING_PROVIDER=local EMBEDDING_MODEL=lsa-256.")


def get_chat_model(settings: Settings | None = None):
    """Return a LangChain chat model, or None when running in offline mode."""
    s = settings or get_settings()
    if s.llm_provider == "offline":
        return None
    from langchain.chat_models import init_chat_model

    kwargs: dict = {"temperature": s.llm_temperature, "max_retries": s.llm_max_retries, "timeout": s.llm_timeout_s}
    if s.llm_provider == "openai" and s.llm_model.startswith(("gpt-5", "o")):
        kwargs.pop("temperature")  # reasoning models only accept the default temperature
        if s.llm_reasoning_effort:
            kwargs["reasoning_effort"] = s.llm_reasoning_effort
    return init_chat_model(s.llm_model, model_provider=s.llm_provider, **kwargs)


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    p = s.embedding_provider
    if p == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=s.embedding_model)
    if p == "google_genai":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=s.embedding_model)
    if p == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings  # pip install langchain-huggingface
        return HuggingFaceEmbeddings(model_name=s.embedding_model, encode_kwargs={"normalize_embeddings": True})
    if p == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=s.embedding_model)
    return LocalLSAEmbeddings(s.index_dir / "lsa_embedder.joblib")


class LocalLSAEmbeddings(Embeddings):
    """Offline embedder: TF-IDF (Thai word tokens + character n-grams) -> SVD.

    Not a substitute for a neural multilingual embedder; it exists so the whole
    pipeline (indexing, Chroma, hybrid retrieval, tests, CI) runs with zero
    network access and zero API keys. It must be fitted on the corpus first.
    """

    def __init__(self, path: Path, dim: int = 256):
        self.path = Path(path)
        self.dim = dim
        self._pipe = None

    # -- fitting / persistence ---------------------------------------------------
    def fit(self, texts: list[str]) -> "LocalLSAEmbeddings":
        import joblib
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion, make_pipeline
        from sklearn.preprocessing import Normalizer

        from .ingestion.thai_text import tokenize

        words = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, ngram_range=(1, 2),
                                sublinear_tf=True, min_df=1)
        chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, min_df=1)
        dim = max(2, min(self.dim, len(texts) - 1))
        self._pipe = make_pipeline(FeatureUnion([("w", words), ("c", chars)]),
                                   TruncatedSVD(n_components=dim, random_state=0), Normalizer(copy=False))
        self._pipe.fit(texts)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipe, self.path)
        return self

    def _load(self):
        if self._pipe is None:
            import joblib
            if not self.path.exists():
                raise FileNotFoundError(f"Local embedder not fitted yet ({self.path}). Run the index build first.")
            self._pipe = joblib.load(self.path)
        return self._pipe

    # -- Embeddings interface ------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._load().transform(texts)
        return np.asarray(vecs, dtype=np.float32).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
