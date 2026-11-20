"""Run the RAG pipeline on every question and save one prediction row per question.

    python src/evaluate.py                                  # flan-t5-small, top-3 retrieval
    python src/evaluate.py --gen-model google/flan-t5-base --out results_base/predictions.json
    python src/evaluate.py --prompt fewshot --out results_fewshot/predictions.json

Each row stores the answer, three confidence scores, whether the model said "I don't know",
whether the answer is correct, and whether the entity's own document was retrieved.
The analysis (tables and figures) runs automatically afterwards; see analyze.py.
"""

from __future__ import annotations

import argparse
import os
import re
import time

from tqdm import tqdm

from confidence import get_confidence
from metrics import is_correct, is_idk
from utils import load_qa_data, save_json

CONF_METHODS = ("max_prob", "entropy")


def entity_doc(item: dict) -> str | None:
    """Document that holds the question's fact (None for unanswerable questions)."""
    if "doc" in item:
        return item["doc"]
    prefix = "india" if item["group"] == "indian" else "world"
    slug = re.sub(r"[^a-z0-9]+", "_", item["entity"].lower()).strip("_")
    return f"{prefix}__{slug}.txt"


def run_evaluation(
    qa_path: str = "data/qa_eval.json",
    output_path: str = "results/predictions.json",
    gen_model: str = "google/flan-t5-small",
    top_k: int = 3,
    prompt_style: str = "default",
    analyze: bool = True,
    abstain_threshold: float = 0.55,
    **_ignored,
):
    from rag_pipeline import LightweightRAG  # heavy imports only when inference is needed

    qa = load_qa_data(qa_path)
    print(f"{len(qa)} questions loaded from {qa_path}")
    rag = LightweightRAG(gen_model_name=gen_model, top_k=top_k, prompt_style=prompt_style)

    rows, t0 = [], time.time()
    for item in tqdm(qa):
        out = rag.answer(item["question"])
        pred = out["answer"].strip()
        retrieved = [r["doc_id"] for r in out.get("retrieved", [])]
        conf = {m: float(get_confidence(m, out)) for m in CONF_METHODS}
        conf["retrieval"] = float(out["retrieved"][0]["score"]) if out.get("retrieved") else 0.0
        rows.append({
            "id": item["id"],
            "group": item["group"],
            "category": item.get("category", "unknown"),
            "entity": item.get("entity"),
            "answerable": bool(item.get("answerable", True)),
            "question": item["question"],
            "answer": item["answer"],
            "aliases": item.get("aliases", []),
            "prediction": pred,
            "said_idk": is_idk(pred),
            "correct": is_correct(pred, item["answer"], item.get("aliases", [])),
            "variant": bool(item.get("variant", False)),
            "own_doc_retrieved": (entity_doc(item) in retrieved) if entity_doc(item) else None,
            "retrieved": retrieved,
            "contexts": out.get("contexts", []),
            **{f"conf_{k}": v for k, v in conf.items()},
        })

    meta = {"gen_model": gen_model, "top_k": top_k, "prompt_style": prompt_style, "n": len(rows),
            "runtime_s": round(time.time() - t0)}
    save_json({"meta": meta, "predictions": rows}, output_path)
    print(f"Saved {len(rows)} predictions to {output_path} ({meta['runtime_s']}s)")

    if analyze:
        from analyze import main as analyze_main

        return analyze_main(output_path, os.path.dirname(output_path) or "results", abstain_threshold)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/qa_eval.json")
    ap.add_argument("--out", default="results/predictions.json")
    ap.add_argument("--gen-model", default="google/flan-t5-small")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--prompt", default="default", choices=["default", "fewshot"])
    ap.add_argument("--threshold", type=float, default=0.55)
    a = ap.parse_args()
    if not os.path.exists(a.qa):
        raise SystemExit(f"{a.qa} not found. Run `python src/make_data.py` first.")
    run_evaluation(a.qa, a.out, a.gen_model, a.top_k, a.prompt, abstain_threshold=a.threshold)
