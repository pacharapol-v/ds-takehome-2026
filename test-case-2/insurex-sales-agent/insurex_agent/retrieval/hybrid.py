"""Hybrid retrieval: dense (Chroma) + lexical (BM25, Thai-tokenised) fused with RRF,
then lifted from child chunks to parent sections ("small-to-big").

Why hybrid: Thai insurance questions are full of exact tokens that dense
embeddings blur - product codes (90/10 vs 90/20), amounts, "ขั้นอาชีพ 1".
BM25 nails those; dense handles paraphrases ("ตายจากอุบัติเหตุ" vs
"เสียชีวิตจากอุบัติเหตุ"). Reciprocal Rank Fusion needs no score calibration
between the two systems.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..config import Settings, get_settings
from ..ingestion.catalog import PRODUCT_BY_ID, detect_products, expand_query
from ..ingestion.thai_text import char_ngrams, tokenize
from .store import KnowledgeBase

log = logging.getLogger(__name__)

# query keywords -> section types worth a small boost (metadata-aware ranking)
SECTION_HINTS: list[tuple[re.Pattern, tuple[str, ...]]] = [
    (re.compile(r"เบี้ย|ราคา|จ่ายเดือนละ|จ่ายปีละ|premium|cost|price", re.I), ("premium_example", "benefit_example")),
    (re.compile(r"อายุรับ|รับประกันถึงอายุ|อายุเท่าไหร่ถึงทำได้|ขั้นต่ำ|ทุนขั้นต่ำ|งวด|ชำระแบบ|เงื่อนไข|ต้องแนบ|eligib|minimum", re.I),
     ("underwriting",)),
    (re.compile(r"เสียชีวิต|มะเร็ง|โรคร้าย|อุบัติเหตุ|เงินคืน|ครบสัญญา|ครบกำหนด|ผลประโยชน์|benefit|payout", re.I),
     ("benefit_example", "benefits")),
    (re.compile(r"ภาษี|ลดหย่อน|tax", re.I), ("overview", "notes_legal")),
    (re.compile(r"บริษัท|รับประกันภัยโดย|ติดต่อ|call center|contact|นายหน้า", re.I), ("notes_legal",)),
]


@dataclass
class Hit:
    chunk_id: str
    text: str
    score: float
    metadata: dict


@dataclass
class RetrievedParent:
    parent_id: str
    text: str
    metadata: dict
    score: float
    hits: list[Hit] = field(default_factory=list)

    def citation(self) -> str:
        m = self.metadata
        return f"{m['product_name']} – {m['section_title']} (ไฟล์ {m['source']} หน้า {m['page']})"


@dataclass
class RetrievalResult:
    query: str
    product_filter: list[str]
    parents: list[RetrievedParent]
    debug: dict = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase | None = None, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.kb = kb or KnowledgeBase(self.s)

    # -- individual retrievers -------------------------------------------------------
    def _dense(self, query: str, k: int, products: list[str]) -> list[tuple[str, float]]:
        flt = {"product_id": {"$in": products}} if products else None
        try:
            res = self.kb.vectorstore.similarity_search_with_score(query, k=k, filter=flt)
        except Exception as e:  # embedding API down -> degrade to BM25-only instead of failing the turn
            log.warning("dense retrieval failed (%s); continuing with BM25 only", e)
            return []
        return [(d.metadata["chunk_id"], 1.0 - float(dist)) for d, dist in res]

    def _bm25(self, query: str, k: int, products: list[str], char_level: bool = False) -> list[tuple[str, float]]:
        q = char_ngrams(query) if char_level else tokenize(query)
        if not q:
            return []
        scores = (self.kb.bm25_char if char_level else self.kb.bm25).get_scores(q)
        if products:
            mask = np.array([c.metadata["product_id"] in products for c in self.kb.children])
            scores = np.where(mask, scores, -1.0)
        order = np.argsort(-scores)[:k]
        return [(self.kb.children[i].chunk_id, float(scores[i])) for i in order if scores[i] > 0]

    # -- fusion ----------------------------------------------------------------------
    def retrieve(self, query: str, products: list[str] | None = None, top_parents: int | None = None,
                 use_filter: bool = True, components: tuple[str, ...] = ("dense", "bm25", "char"),
                 boosts: bool | None = None) -> RetrievalResult:
        products = [p for p in (products or []) if p in PRODUCT_BY_ID]
        if not products:
            products = detect_products(query)
        flt = products if use_filter else []
        expanded = expand_query(query)                      # colloquial -> brochure vocabulary
        dense = self._dense(query, self.s.retrieval_k_dense, flt) if "dense" in components else []
        lexical = self._bm25(expanded, self.s.retrieval_k_bm25, flt) if "bm25" in components else []
        lexical_char = (self._bm25(expanded, self.s.retrieval_k_bm25, flt, char_level=True)
                        if "char" in components else [])

        rrf: dict[str, float] = defaultdict(float)
        for ranked in (dense, lexical, lexical_char):
            for rank, (cid, _) in enumerate(ranked):
                rrf[cid] += 1.0 / (self.s.rrf_k + rank + 1)

        boosts = self.s.section_boosts if boosts is None else boosts
        boosted_sections = {sec for pat, secs in SECTION_HINTS if pat.search(query) for sec in secs} if boosts else set()
        for cid in list(rrf):
            meta = self.kb.child_by_id[cid].metadata
            if meta["section"] in boosted_sections:
                rrf[cid] *= 1.15
            if products and not use_filter and meta["product_id"] in products:
                rrf[cid] *= 1.25  # soft preference when the hard filter is off

        # lift children -> parents
        by_parent: dict[str, list[Hit]] = defaultdict(list)
        for cid, score in sorted(rrf.items(), key=lambda x: -x[1]):
            c = self.kb.child_by_id[cid]
            by_parent[c.parent_id].append(Hit(cid, c.text, score, c.metadata))
        ranked_parents = []
        for pid, hits in by_parent.items():
            p = self.kb.parents[pid]
            # best child dominates; extra matching children add a small bonus
            score = hits[0].score + 0.25 * sum(h.score for h in hits[1:3])
            ranked_parents.append(RetrievedParent(pid, p.text, p.metadata, score, hits[:3]))
        ranked_parents.sort(key=lambda r: -r.score)

        limit = top_parents or self.s.max_context_parents
        # comparison questions: make sure every named product is represented
        if len(products) > 1:
            limit = max(limit, len(products) + 1)
            selected, seen = [], set()
            for prod in products:
                best = next((r for r in ranked_parents if r.metadata["product_id"] == prod), None)
                if best:
                    selected.append(best)
                    seen.add(best.parent_id)
            selected += [r for r in ranked_parents if r.parent_id not in seen]
            ranked_parents = selected
        result = RetrievalResult(query, flt, ranked_parents[:limit],
                                 debug={"dense": dense[:5], "bm25": lexical[:5], "bm25_char": lexical_char[:5],
                                        "expanded_query": expanded, "boosted": sorted(boosted_sections)})
        return result

    def child_hits(self, query: str, k: int = 10, products: list[str] | None = None) -> list[Hit]:
        """Flat child-level ranking (used by the retrieval evaluation)."""
        res = self.retrieve(query, products=products, top_parents=50)
        hits = [h for p in res.parents for h in p.hits]
        return sorted(hits, key=lambda h: -h.score)[:k]
