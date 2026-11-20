"""Tests for answer matching, metrics and the analysis pipeline (no model downloads needed)."""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from metrics import ece, is_correct, is_idk, normalize, risk_coverage  # noqa: E402


def test_aliases_and_accents():
    assert is_correct("The capital is Bangalore.", "Bengaluru", ["Bangalore"])
    assert is_correct("Gabriel García Márquez", "Gabriel Garcia Marquez")
    assert is_correct("st. paul", "Saint Paul", ["St. Paul"])


def test_partial_prediction_is_not_correct():
    # the original check accepted pred-in-gold, so "india" would match "New Delhi, India"-style golds
    assert not is_correct("india", "Uttar Pradesh")
    assert not is_correct("the", "The God of Small Things")
    assert not is_correct("Pradesh", "Uttar Pradesh")


def test_whole_word_matching():
    assert not is_correct("Kotagiri", "Kota")
    assert is_correct("Kota, Rajasthan", "Kota")


def test_idk_detection():
    for s in ["I don't know", "i dont know.", "The context is insufficient.", "", "unknown"]:
        assert is_idk(s)
    assert not is_idk("Mumbai")
    assert not is_correct("I don't know", "Mumbai")


def test_normalize():
    assert normalize("  The  Taj-Mahal! ") == "taj mahal"


def test_ece_and_risk_coverage():
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)
    assert ece(p, y) < 0.02
    assert ece(np.ones(100), np.zeros(100)) == 1.0
    cov, risk, aurc = risk_coverage(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0]))
    assert np.allclose(cov, [1 / 3, 2 / 3, 1]) and np.allclose(risk, [0, 0, 1 / 3])
    assert abs(aurc - 1 / 9) < 1e-9


def test_analysis_runs_end_to_end(tmp_path):
    from analyze import main

    rng = np.random.default_rng(1)
    rows = []
    for i in range(200):
        g = "indian" if i % 2 else "non_indian"
        ans = (i // 2) % 4 != 0
        idk = (not ans and rng.random() < 0.4) or (ans and rng.random() < 0.05)
        correct = ans and not idk and rng.random() < (0.6 if g == "indian" else 0.8)
        rows.append({"id": str(i), "group": g, "category": ["a", "b"][i % 3 == 0], "answerable": ans,
                     "said_idk": idk, "correct": correct, "own_doc_retrieved": ans and rng.random() < 0.9,
                     "conf_max_prob": float(rng.random()), "conf_entropy": float(rng.random()),
                     "conf_retrieval": float(rng.random())})
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps({"meta": {}, "predictions": rows}))
    rep = main(str(p), str(tmp_path))
    assert (tmp_path / "fig1_reliability_risk_coverage.png").exists()
    assert (tmp_path / "group_gaps.csv").exists()
    s = {r["group"]: r for r in rep["summary"]}
    assert s["all"]["questions"] == 200 and s["indian"]["unanswerable"] == 25


def test_mock_pipeline_end_to_end(tmp_path):
    """Run the real data through a toy retriever and a toy generator that always answers, then analyse.

    The toy generator copies the answer of the best-matching retrieved sentence, which is the failure
    mode seen with flan-t5-small. The grounding check must catch it on unanswerable questions.
    """
    import re

    from analyze import main
    from metrics import grounded, is_correct, is_idk, normalize

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qa = json.load(open(os.path.join(root, "data", "qa_eval.json")))
    docs = {f: open(os.path.join(root, "data", "documents", f)).read()
            for f in os.listdir(os.path.join(root, "data", "documents"))}
    toks = {f: set(normalize(t).split()) for f, t in docs.items()}
    rng = np.random.default_rng(0)
    rows = []
    for q in qa:
        qt = set(normalize(q["question"]).split())
        ranked = sorted(docs, key=lambda f: -len(qt & toks[f]))[:3]
        sents = [s for f in ranked for s in re.split(r"(?<=[a-z]{2}[.!?])\s+", docs[f])]
        best = max(sents, key=lambda s: len(qt & set(normalize(s).split())))
        pred = best.split(" is ")[-1].split(" by ")[-1].rstrip(".")
        rows.append({"id": q["id"], "group": q["group"], "category": q["category"], "answerable": q["answerable"],
                     "variant": q["variant"], "question": q["question"], "answer": q["answer"], "prediction": pred,
                     "said_idk": is_idk(pred), "correct": is_correct(pred, q["answer"], q["aliases"]),
                     "own_doc_retrieved": (q["doc"] in ranked) if q["doc"] else None, "retrieved": ranked,
                     "contexts": [docs[f] for f in ranked], "conf_max_prob": float(rng.uniform(0.5, 1)),
                     "conf_entropy": float(rng.uniform(0.3, 1)), "conf_retrieval": float(rng.uniform(0.4, 0.9))})
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps({"meta": {"gen_model": "mock"}, "predictions": rows}))
    rep = main(str(p), str(tmp_path))
    pol = {(r["policy"], r["group"]): r for r in rep["policies"]}
    # the grounding check should block most answers to unanswerable questions...
    assert pol[("grounding check", "all")]["answered_unanswerable"] < 0.2
    # ...while the toy model on its own answers all of them
    assert pol[("model's own 'I don't know' only", "all")]["answered_unanswerable"] == 1.0
    assert (tmp_path / "RESULTS.md").exists() and (tmp_path / "fig3_abstention_policies.png").exists()
    assert grounded("Panaji", "Which city is the capital of Goa, India?", ["The capital of Goa is Panaji."])
