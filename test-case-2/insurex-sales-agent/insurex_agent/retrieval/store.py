"""Index build + load: Chroma (dense) + BM25 (lexical) + parent docstore."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ..config import Settings, get_settings
from ..ingestion.catalog import PRODUCTS
from ..ingestion.chunker import Chunk, Parent, build_chunks, save_processed
from ..ingestion.pdf_parser import parse_pdf
from ..ingestion.thai_text import char_ngrams, tokenize
from ..llm import LocalLSAEmbeddings, get_embeddings

# HNSW is approximate; with a few hundred vectors a large ef makes search effectively
# exact, which also makes the evaluation reproducible between index builds.
COLLECTION_METADATA = {"hnsw:space": "cosine", "hnsw:construction_ef": 200, "hnsw:search_ef": 200, "hnsw:M": 32}

log = logging.getLogger(__name__)


class KnowledgeBase:
    """Read-side handle on the indexed knowledge base."""

    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        idx = self.s.index_dir
        manifest_path = idx / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Knowledge base not built ({manifest_path} missing). Run: python -m insurex_agent.ingestion.build_index"
            )
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest["collection"] != self.s.collection_name:
            raise RuntimeError(
                f"Index was built with '{self.manifest['collection']}' but settings point to "
                f"'{self.s.collection_name}'. Rebuild the index after changing EMBEDDING_PROVIDER/MODEL."
            )
        self.parents: dict[str, Parent] = {
            p["parent_id"]: Parent(**p) for p in json.loads((idx / "parents.json").read_text(encoding="utf-8"))
        }
        self.children: list[Chunk] = [
            Chunk(**json.loads(l)) for l in (idx / "children.jsonl").read_text(encoding="utf-8").splitlines() if l
        ]
        self.child_by_id = {c.chunk_id: c for c in self.children}
        self._tokens = [tokenize(c.text) for c in self.children]
        self.bm25 = BM25Okapi(self._tokens)                                   # word level
        self.bm25_char = BM25Okapi([char_ngrams(c.text) for c in self.children])  # char 3-gram level
        self.embeddings = get_embeddings(self.s)
        self.vectorstore = Chroma(
            collection_name=self.s.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.s.chroma_dir),
            collection_metadata=COLLECTION_METADATA,
        )


def build_index(settings: Settings | None = None, rebuild: bool = True) -> dict:
    """PDF -> markdown -> parent/child chunks -> Chroma + BM25 docstore."""
    s = settings or get_settings()
    t0 = time.time()
    all_parents: list[Parent] = []
    all_children: list[Chunk] = []
    report = {"documents": []}
    for product in PRODUCTS:
        pdf = s.pdf_dir / product.file_name
        if not pdf.exists():
            raise FileNotFoundError(f"Missing knowledge-base PDF: {pdf}")
        parsed = parse_pdf(pdf)
        parents, children = build_chunks(parsed)
        save_processed(parsed, parents, children, s.processed_dir)
        issues = sum(p.quality_issues for p in parsed.pages)
        if issues:
            log.warning("%s: %d Thai glyph issues detected after extraction", pdf.name, issues)
        report["documents"].append({
            "file": pdf.name, "product_id": product.product_id, "pages": len(parsed.pages),
            "parents": len(parents), "children": len(children), "thai_quality_issues": issues,
        })
        all_parents += parents
        all_children += children

    idx = s.index_dir
    if rebuild and s.chroma_dir.exists():
        shutil.rmtree(s.chroma_dir)
    idx.mkdir(parents=True, exist_ok=True)

    if s.embedding_provider == "local":
        LocalLSAEmbeddings(idx / "lsa_embedder.joblib").fit([c.text for c in all_children])
    embeddings = get_embeddings(s)
    vs = Chroma(collection_name=s.collection_name, embedding_function=embeddings,
                persist_directory=str(s.chroma_dir), collection_metadata=COLLECTION_METADATA)
    docs = [Document(page_content=c.text, metadata={**c.metadata, "parent_id": c.parent_id, "chunk_id": c.chunk_id})
            for c in all_children]
    batch = 64
    for i in range(0, len(docs), batch):
        vs.add_documents(docs[i:i + batch], ids=[d.metadata["chunk_id"] for d in docs[i:i + batch]])

    (idx / "parents.json").write_text(json.dumps([asdict(p) for p in all_parents], ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    with open(idx / "children.jsonl", "w", encoding="utf-8") as f:
        for c in all_children:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    report.update({
        "collection": s.collection_name,
        "embedding_provider": s.embedding_provider,
        "embedding_model": s.embedding_model,
        "n_parents": len(all_parents),
        "n_children": len(all_children),
        "seconds": round(time.time() - t0, 1),
    })
    (idx / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
