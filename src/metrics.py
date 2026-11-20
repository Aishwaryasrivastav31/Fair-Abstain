"""Answer matching, abstention detection and calibration / selective-prediction metrics.

Pure NumPy, no model dependencies, so it can be unit-tested without downloading anything.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

IDK_PATTERNS = (
    "i don't know", "i dont know", "i do not know", "don't know", "dont know", "unknown",
    "not mentioned", "not provided", "not stated", "not given", "insufficient", "cannot be determined",
    "can't be determined", "no answer", "unanswerable", "not enough information", "no information",
)  # fmt: skip


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, drop articles, collapse spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_idk(pred: str) -> bool:
    """True if the model declined to answer."""
    p = pred.lower().replace("’", "'").strip()
    return not normalize(p) or any(k in p for k in IDK_PATTERNS)


def is_correct(pred: str, answer: str, aliases: list[str] | None = None) -> bool:
    """True if the gold answer or any alias appears in the prediction as whole words.

    The original check also accepted `pred in gold`, which marked short or partial outputs
    ("india", "the") as correct; that direction is not used.
    """
    if is_idk(pred):
        return False
    p = f" {normalize(pred)} "
    for gold in [answer, *(aliases or [])]:
        g = normalize(gold)
        if g and f" {g} " in p:
            return True
    return False


def ece(conf: np.ndarray, ok: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    if len(conf) == 0:
        return float("nan")
    idx = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def reliability_bins(conf: np.ndarray, ok: np.ndarray, n_bins: int = 10):
    """(mean confidence, observed rate, count) per non-empty equal-width bin."""
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    idx = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append((conf[m].mean(), ok[m].mean(), int(m.sum())))
    return rows


def risk_coverage(conf: np.ndarray, ok: np.ndarray):
    """Coverage and selective risk when answering only the k most confident questions, k = 1..n.

    Returns coverage (k/n), risk (error rate among answered) and AURC (mean risk over coverages).
    """
    conf, ok = np.asarray(conf, float), np.asarray(ok, float)
    n = len(conf)
    if n == 0:
        return np.array([]), np.array([]), float("nan")
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - ok[order]
    k = np.arange(1, n + 1)
    risk = np.cumsum(err) / k
    return k / n, risk, float(risk.mean())


def bootstrap_diff(fn, a: dict, b: dict, n: int = 1000, seed: int = 2026):
    """Bootstrap 95% CI and two-sided p-value for fn(a) - fn(b), resampling questions within each group.

    a, b: dicts of equal-length arrays (one entry per question).
    The p-value is twice the smaller tail share of bootstrap differences on either side of zero.
    """
    rng = np.random.default_rng(seed)
    na, nb = len(next(iter(a.values()))), len(next(iter(b.values())))
    point = fn(a) - fn(b)
    diffs = []
    for _ in range(n):
        ia, ib = rng.integers(0, na, na), rng.integers(0, nb, nb)
        diffs.append(fn({k: v[ia] for k, v in a.items()}) - fn({k: v[ib] for k, v in b.items()}))
    diffs = np.asarray(diffs, float)
    diffs = diffs[~np.isnan(diffs)]
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return float(point), float(lo), float(hi), float(p)


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


QUESTION_STOP = set("""
which what who whom where when is are was were the a an of in on to for and does do did by from at it its
city state states india indian us usa located capital main official language known playing play plays sport
musical instrument wrote author written directed director film started founded home mountain peak there
""".split())


def key_terms(question: str) -> set[str]:
    """Content words of the question: the entity it asks about, once template words are removed."""
    return {t for t in normalize(question).split() if t not in QUESTION_STOP and len(t) > 1}


def _sentences(text: str) -> list[str]:
    # split after a full stop that ends a lower-case word, so "R. K. Narayan" and "St. Louis" stay whole
    return [x for x in re.split(r"(?<=[a-z]{2}[.!?])\s+", text) if x.strip()]


def grounded(pred: str, question: str, contexts: list[str], min_overlap: float = 0.6,
             min_answer_overlap: float = 0.8) -> bool:
    """Lexical support check, using only the question, the answer and the retrieved text.

    Some retrieved sentence must contain at least min_answer_overlap of the answer's words, and that
    sentence together with the one before it must contain at least min_overlap of the question's key
    terms. A model that copies a fact about a different entity (the typical failure on unanswerable
    questions) fails the check.
    """
    if is_idk(pred):
        return False
    ans = {t for t in normalize(pred).split() if t not in QUESTION_STOP}
    if not ans:
        return False
    terms = key_terms(question)
    for ctx in contexts:
        sents = _sentences(ctx)
        for i, sent in enumerate(sents):
            words = set(normalize(sent).split())
            if len(ans & words) / len(ans) < min_answer_overlap:
                continue
            window = set(normalize(" ".join(sents[max(0, i - 1): i + 1])).split())
            if not terms or len(terms & window) / len(terms) >= min_overlap:
                return True
    return False
