"""Shared fixtures. Everything runs offline: rule-based brain + local LSA embeddings,
so the suite needs no API key and no network."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from insurex_agent.config import Settings  # noqa: E402


@pytest.fixture(scope="session")
def offline_settings(tmp_path_factory) -> Settings:
    base = tmp_path_factory.mktemp("kb")
    s = Settings(
        llm_provider="offline", embedding_provider="local", embedding_model="lsa-256",
        index_dir=base / "index", processed_dir=base / "processed", state_dir=base / "state",
        leads_backend="direct", _env_file=None,
    )
    s.ensure_dirs()
    from insurex_agent.retrieval.store import build_index

    build_index(s)
    return s


@pytest.fixture(scope="session")
def retriever(offline_settings):
    from insurex_agent.retrieval.hybrid import HybridRetriever

    return HybridRetriever(settings=offline_settings)


@pytest.fixture()
def fresh_state(offline_settings, tmp_path) -> Settings:
    """Same index, but a clean state dir (leads / checkpoints / sessions) per test."""
    s = offline_settings.model_copy(update={"state_dir": tmp_path / "state"})
    s.ensure_dirs()
    return s
