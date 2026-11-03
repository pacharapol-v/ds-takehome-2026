"""Retrieval evaluation + ablation (no LLM needed).

    python scripts/eval_retrieval.py            # writes eval/results/retrieval_<collection>.md/json

Gold standard: eval/qa_set.jsonl - every answerable question lists the product(s)
and the literal fact strings (from the brochures) the answer requires. A retrieved
parent section counts as *relevant* if it belongs to an expected product and
contains the required facts.

Metrics (top-k parent sections, k = what the agent puts in the prompt = 4):
  Hit@1 / Hit@k   - a relevant section is ranked 1st / within k
  MRR             - mean reciprocal rank of the first relevant section
  ProductP@k      - share of the k sections that belong to the asked product(s)
                    (precision w.r.t. product; noise from other products is what makes
                    LLMs mix up 90/10 vs 90/20 numbers)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from insurex_agent.config import get_settings  # noqa: E402
from insurex_agent.retrieval.hybrid import HybridRetriever  # noqa: E402

K = 4
CONFIGS = {
    "dense only": dict(components=("dense",), boosts=False),
    "BM25 word only": dict(components=("bm25",), boosts=False),
    "BM25 char-3gram only": dict(components=("char",), boosts=False),
    "hybrid (RRF)": dict(components=("dense", "bm25", "char"), boosts=False),
    "hybrid + section boosts": dict(components=("dense", "bm25", "char"), boosts=True),
    "hybrid + product filter": dict(components=("dense", "bm25", "char"), boosts=False, use_filter=True),
    "hybrid + boosts + product filter": dict(components=("dense", "bm25", "char"), boosts=True, use_filter=True),
}


def norm(s: str) -> str:
    return re.sub(r"[\s,]", "", s)


def is_relevant(parent, q) -> bool:
    if q["products"] and parent.metadata["product_id"] not in q["products"]:
        return False
    text = norm(parent.text + " " + " ".join(h.text for h in parent.hits))
    if q["facts_all"] and not all(norm(f) in text for f in q["facts_all"]):
        return False
    if q["facts_any"] and not any(norm(f) in text for f in q["facts_any"]):
        return False
    return True


def evaluate(retriever: HybridRetriever, questions: list[dict], cfg: dict) -> dict:
    cfg = dict(cfg)
    use_filter = cfg.pop("use_filter", False)
    hit1 = hitk = rr = pp = 0.0
    misses = []
    for q in questions:
        res = retriever.retrieve(q["question"], products=None, top_parents=K, use_filter=use_filter, **cfg)
        parents = res.parents[:K]
        ranks = [i for i, p in enumerate(parents, 1) if is_relevant(p, q)]
        if ranks:
            hit1 += ranks[0] == 1
            hitk += 1
            rr += 1 / ranks[0]
        else:
            misses.append(q["id"])
        if q["products"]:
            pp += sum(p.metadata["product_id"] in q["products"] for p in parents) / max(1, len(parents))
        else:
            pp += 1.0
    n = len(questions)
    return {"Hit@1": hit1 / n, f"Hit@{K}": hitk / n, "MRR": rr / n, f"ProductP@{K}": pp / n, "misses": misses}


def main() -> None:
    s = get_settings()
    qs = [json.loads(l) for l in (ROOT / "eval" / "qa_set.jsonl").read_text(encoding="utf-8").splitlines() if l]
    answerable = [q for q in qs if q["answerable"]]
    retriever = HybridRetriever(settings=s)
    rows = {name: evaluate(retriever, answerable, cfg) for name, cfg in CONFIGS.items()}

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = s.collection_name
    lines = [f"# Retrieval evaluation — embeddings `{s.embedding_provider}:{s.embedding_model}`", "",
             f"{len(answerable)} answerable questions (Thai + English), top-{K} parent sections.", "",
             f"| Config | Hit@1 | Hit@{K} | MRR | ProductP@{K} | misses |", "|---|---|---|---|---|---|"]
    for name, r in rows.items():
        lines.append(f"| {name} | {r['Hit@1']:.2f} | {r[f'Hit@{K}']:.2f} | {r['MRR']:.2f} | {r[f'ProductP@{K}']:.2f} | "
                     f"{', '.join(r['misses']) or '-'} |")
    md = "\n".join(lines) + "\n"
    (out_dir / f"retrieval_{tag}.md").write_text(md, encoding="utf-8")
    (out_dir / f"retrieval_{tag}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
