"""Tables, figures and RESULTS.md from results/predictions.json.

    python src/analyze.py [results/predictions.json] [results]

Main analysis uses the 628 main questions; the spelling-variant questions are analysed separately.

Definitions
-----------
- answered: the model gave an answer (did not say "I don't know").
- accuracy: share of answerable questions answered correctly.
- missed abstention: share of unanswerable questions that got an answer anyway.
- false abstention: share of answerable questions where the model said "I don't know".
- ECE / overconfidence: on answered questions, confidence against correctness of the answer.
- risk-coverage / AURC: answered questions ranked by confidence; AURC is the mean error rate over the
  coverage levels reachable (lower is better).
- abstention policies: rules for when the system should answer. Thresholds are chosen by 5-fold
  cross-validation over questions (chosen on four folds, applied to the fifth), maximising
  utility = (+1 per correct answer, -1 per wrong answer, 0 per abstention) / questions.
- grounding check: answer only if the answer appears in a retrieved sentence that also mentions the
  question's subject (metrics.grounded); uses no labels.
- calibration: Platt scaling and isotonic regression fitted by 5-fold cross-validation, either on both
  groups together (pooled) or separately per group.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from metrics import bootstrap_diff, ece, grounded, holm, reliability_bins, risk_coverage

GROUPS = ("indian", "non_indian")
LABEL = {"indian": "Indian entities", "non_indian": "Non-Indian entities"}
COLOR = {"indian": "#c2410c", "non_indian": "#1d4ed8"}
METHODS = ("max_prob", "entropy", "retrieval")
TAUS = np.round(np.arange(0.30, 0.96, 0.05), 2)
SEED = 2026
K_FOLDS = 5


# ----------------------------------------------------------------------------- loading
def load(path: str) -> tuple[pd.DataFrame, dict]:
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    df = pd.DataFrame(blob["predictions"])
    if "variant" not in df:
        df["variant"] = False
    if "contexts" in df:
        df["grounded"] = [grounded(p, q, c) for p, q, c in zip(df.prediction, df.question, df.contexts, strict=True)]
    else:
        df["grounded"] = np.nan
    return df, blob.get("meta", {})


def _arr(d: pd.DataFrame, m: str) -> dict:
    return {
        "conf": d[f"conf_{m}"].to_numpy(float),
        "correct": d.correct.to_numpy(float),
        "idk": d.said_idk.to_numpy(bool),
        "ans": d.answerable.to_numpy(bool),
    }


# ----------------------------------------------------------------------------- core metrics
def accuracy(a):
    return a["correct"][a["ans"]].mean() if a["ans"].any() else np.nan


def hallucination(a):
    u = ~a["ans"]
    return (~a["idk"][u]).mean() if u.any() else np.nan


def ece_answered(a):
    m = ~a["idk"]
    return ece(a["conf"][m], a["correct"][m]) if m.any() else np.nan


def overconf(a):
    m = ~a["idk"]
    return a["conf"][m].mean() - a["correct"][m].mean() if m.any() else np.nan


def rc(a):
    n = len(a["conf"])
    m = ~a["idk"]
    cov, risk, _ = risk_coverage(a["conf"][m], a["correct"][m])
    return cov * m.sum() / n, risk, (float(risk.mean()) if len(risk) else np.nan)


def aurc(a):
    return rc(a)[2]


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in ("all", *GROUPS):
        d = df if g == "all" else df[df.group == g]
        ans, una = d[d.answerable], d[~d.answerable]
        base = {
            "group": g, "questions": len(d), "answerable": len(ans), "unanswerable": len(una),
            "accuracy_answerable": ans.correct.mean(),
            "false_abstention": ans.said_idk.mean(),
            "missed_abstention_unanswerable": (~una.said_idk).mean() if len(una) else np.nan,
            "own_doc_retrieved": ans.own_doc_retrieved.astype(float).mean(),
            "accuracy_when_own_doc_retrieved": ans[ans.own_doc_retrieved == True].correct.mean(),  # noqa: E712
            "grounded_share_answerable": ans.grounded.astype(float).mean(),
            "grounded_share_unanswerable": una.grounded.astype(float).mean() if len(una) else np.nan,
        }
        for m in METHODS:
            a = _arr(d, m)
            base[f"ece_{m}"] = ece_answered(a)
            base[f"overconfidence_{m}"] = overconf(a)
            base[f"aurc_{m}"] = aurc(a)
        rows.append(base)
    return pd.DataFrame(rows)


def gap_table(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Indian minus non-Indian, bootstrap 95% CI and p-value, Holm-adjusted across the family."""
    di, dn = df[df.group == "indian"], df[df.group == "non_indian"]
    rows = []
    for name, fn, m in [
        ("accuracy_answerable", accuracy, "max_prob"),
        ("missed_abstention_unanswerable", hallucination, "max_prob"),
        *[(f"ece_{m}", ece_answered, m) for m in METHODS],
        *[(f"overconfidence_{m}", overconf, m) for m in METHODS],
        *[(f"aurc_{m}", aurc, m) for m in METHODS],
    ]:
        p, lo, hi, pv = bootstrap_diff(fn, _arr(di, m), _arr(dn, m), n=n_boot)
        rows.append({"metric": name, "indian_minus_non_indian": p, "ci_low": lo, "ci_high": hi, "p": pv})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p)
    out["significant_after_holm"] = out.p_holm < 0.05
    return out


def category_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (c, g), d in df.groupby(["category", "group"]):
        ans, una = d[d.answerable], d[~d.answerable]
        rows.append({"category": c, "group": g, "answerable": len(ans), "unanswerable": len(una),
                     "accuracy_answerable": ans.correct.mean(),
                     "missed_abstention_unanswerable": (~una.said_idk).mean() if len(una) else np.nan,
                     "own_doc_retrieved": ans.own_doc_retrieved.astype(float).mean()})
    return pd.DataFrame(rows)


def threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        for g in GROUPS:
            a = _arr(df[df.group == g], m)
            for t in TAUS:
                keep = (~a["idk"]) & (a["conf"] >= t)
                rows.append({"method": m, "group": g, "tau": t, "coverage": keep.mean(),
                             "selective_accuracy": a["correct"][keep].mean() if keep.any() else np.nan,
                             "unanswerable_abstained": (~keep[~a["ans"]]).mean() if (~a["ans"]).any() else np.nan})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- abstention policies
def _folds(df: pd.DataFrame) -> np.ndarray:
    """Fold id per question, stratified by group and answerability."""
    rng = np.random.default_rng(SEED)
    fold = np.empty(len(df), int)
    for _, idx in df.groupby(["group", "answerable"]).indices.items():
        idx = rng.permutation(idx)
        fold[idx] = np.arange(len(idx)) % K_FOLDS
    return fold


def _utility(keep, correct):
    return float((keep & correct).sum() - (keep & ~correct).sum()) / max(len(keep), 1)


def _best_tau(conf, base_keep, correct):
    cands = np.r_[-np.inf, np.unique(conf[base_keep])]
    best, best_u = -np.inf, -np.inf
    for t in cands:
        u = _utility(base_keep & (conf >= t), correct)
        if u > best_u + 1e-12:
            best, best_u = t, u
    return best


def policies(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each policy decides per question whether to answer. Returns (per-group results, kept masks)."""
    fold = _folds(df)
    correct = df.correct.to_numpy(bool)
    idk = df.said_idk.to_numpy(bool)
    grp = df.group.to_numpy()
    has_ground = not df.grounded.isna().all()
    ground = df.grounded.fillna(False).to_numpy(bool)
    conf = {m: df[f"conf_{m}"].to_numpy(float) for m in METHODS}

    def cv_threshold(base, c, per_group=False):
        keep = np.zeros(len(df), bool)
        for k in range(K_FOLDS):
            tr, te = fold != k, fold == k
            if per_group:
                for g in GROUPS:
                    trg, teg = tr & (grp == g), te & (grp == g)
                    t = _best_tau(c[trg], base[trg], correct[trg])
                    keep[teg] = base[teg] & (c[teg] >= t)
            else:
                t = _best_tau(c[tr], base[tr], correct[tr])
                keep[te] = base[te] & (c[te] >= t)
        return keep

    P = {"model's own 'I don't know' only": ~idk,
         "threshold on max_prob": cv_threshold(~idk, conf["max_prob"]),
         "threshold on max_prob, per-group threshold": cv_threshold(~idk, conf["max_prob"], True),
         "threshold on retrieval score": cv_threshold(~idk, conf["retrieval"]),
         "threshold on retrieval score, per-group threshold": cv_threshold(~idk, conf["retrieval"], True)}
    if has_ground:
        P["grounding check"] = ground
        P["grounding check + threshold on max_prob"] = cv_threshold(ground, conf["max_prob"])
    rows = []
    ans = df.answerable.to_numpy(bool)
    for name, keep in P.items():
        for g in ("all", *GROUPS):
            m = np.ones(len(df), bool) if g == "all" else grp == g
            k, c, a = keep[m], correct[m], ans[m]
            rows.append({"policy": name, "group": g, "coverage": k.mean(),
                         "selective_accuracy": c[k].mean() if k.any() else np.nan,
                         "answered_unanswerable": k[~a].mean() if (~a).any() else np.nan,
                         "correct_answers_kept": (k & c)[a].sum() / max((c & a).sum(), 1),
                         "utility": _utility(k, c)})
    return pd.DataFrame(rows), pd.DataFrame(P)


def policy_gaps(df: pd.DataFrame, masks: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """Indian minus non-Indian for each policy: answered-unanswerable rate and selective accuracy."""
    rows = []
    for name in masks.columns:
        keep = masks[name].to_numpy(bool)
        pack = {g: {"keep": keep[(df.group == g).to_numpy()], "correct": df.correct[df.group == g].to_numpy(bool),
                    "ans": df.answerable[df.group == g].to_numpy(bool)} for g in GROUPS}
        def sel_acc(a):
            return a["correct"][a["keep"]].mean() if a["keep"].any() else np.nan

        for metric, fn in (("answered_unanswerable", lambda a: a["keep"][~a["ans"]].mean()),
                           ("selective_accuracy", sel_acc)):
            p, lo, hi, pv = bootstrap_diff(fn, pack["indian"], pack["non_indian"], n=n_boot)
            rows.append({"policy": name, "metric": metric, "indian_minus_non_indian": p, "ci_low": lo, "ci_high": hi,
                         "p": pv})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p)
    return out


# ----------------------------------------------------------------------------- calibration
def calibration(df: pd.DataFrame) -> pd.DataFrame:
    """ECE per group before and after post-hoc calibration (5-fold CV, on answered questions)."""
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    d = df[~df.said_idk].reset_index(drop=True)
    if len(d) < 20:
        return pd.DataFrame()
    fold = _folds(d)
    y = d.correct.to_numpy(int)
    grp = d.group.to_numpy()
    rows = []
    for m in ("max_prob", "retrieval"):
        c = np.clip(d[f"conf_{m}"].to_numpy(float), 1e-4, 1 - 1e-4)
        x = np.log(c / (1 - c)).reshape(-1, 1)

        def fit_predict(kind, tr, te, x=x, c=c):
            if len(np.unique(y[tr])) < 2:
                return np.full(te.sum(), y[tr].mean())
            if kind == "platt":
                return LogisticRegression().fit(x[tr], y[tr]).predict_proba(x[te])[:, 1]
            return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(c[tr], y[tr]).predict(c[te])

        cal = {"uncalibrated": c}
        for kind in ("platt", "isotonic"):
            for scope in ("pooled", "per group"):
                p = np.empty(len(d))
                for k in range(K_FOLDS):
                    tr, te = fold != k, fold == k
                    if scope == "pooled":
                        p[te] = fit_predict(kind, tr, te)
                    else:
                        for g in GROUPS:
                            teg = te & (grp == g)
                            if teg.any():
                                p[teg] = fit_predict(kind, tr & (grp == g), teg)
                cal[f"{kind}, {scope}"] = p
        for name, p in cal.items():
            r = {"confidence": m, "calibration": name}
            for g in GROUPS:
                mg = grp == g
                r[f"ece_{g}"] = ece(p[mg], y[mg])
            r["ece_gap"] = r["ece_indian"] - r["ece_non_indian"]
            pack = {g: {"p": p[grp == g], "y": y[grp == g].astype(float)} for g in GROUPS}
            _, lo, hi, _ = bootstrap_diff(lambda a: ece(a["p"], a["y"]), pack["indian"], pack["non_indian"], n=500)
            r["gap_ci_low"], r["gap_ci_high"] = lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- spelling variants
def variant_table(full: pd.DataFrame) -> pd.DataFrame:
    """Questions that use another spelling or older name, against the same question with the main name."""
    v = full[full.variant]
    if v.empty:
        return pd.DataFrame()
    base = full[~full.variant].set_index("id")
    rows = []
    for g in ("all", *GROUPS):
        vv = v if g == "all" else v[v.group == g]
        if vv.empty:
            continue
        b = base.loc[[i[:-2] for i in vv.id]]
        rows.append({"group": g, "pairs": len(vv),
                     "accuracy_main_name": b.correct.mean(), "accuracy_variant": vv.correct.mean(),
                     "doc_retrieved_main_name": b.own_doc_retrieved.astype(float).mean(),
                     "doc_retrieved_variant": vv.own_doc_retrieved.astype(float).mean(),
                     "grounded_main_name": b.grounded.astype(float).mean(),
                     "grounded_variant": vv.grounded.astype(float).mean()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- figures
def figures(df: pd.DataFrame, pol: pd.DataFrame, cal: pd.DataFrame, out: str, method: str = "max_prob"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e5e7eb",
                         "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="#6b7280", lw=1)
    for g in GROUPS:
        a = _arr(df[df.group == g], method)
        m = ~a["idk"]
        b = reliability_bins(a["conf"][m], a["correct"][m])
        if b:
            x, y, _ = zip(*b, strict=True)
            ax.plot(x, y, "o-", color=COLOR[g], label=f"{LABEL[g]} (ECE {ece_answered(a):.3f})")
    ax.set_xlabel("Confidence (max token probability)")
    ax.set_ylabel("Share of answers that are correct")
    ax.set_title("Reliability of answered questions")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for g in GROUPS:
        cov, risk, au = rc(_arr(df[df.group == g], method))
        ax.plot(cov, risk, color=COLOR[g], label=f"{LABEL[g]} (AURC {au:.3f})")
    ax.set_xlabel("Coverage (share of all questions answered)")
    ax.set_ylabel("Error rate among answered")
    ax.set_title("Risk-coverage")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_reliability_risk_coverage.png"))
    plt.close(fig)

    cat = category_table(df)
    cats = sorted(cat.category.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, col, title in ((axes[0], "accuracy_answerable", "Accuracy on answerable questions"),
                           (axes[1], "missed_abstention_unanswerable", "Answered when it should abstain")):
        for i, g in enumerate(GROUPS):
            v = cat[cat.group == g].set_index("category").reindex(cats)[col]
            ax.barh(np.arange(len(cats)) + (i - 0.5) * 0.38, v, 0.38, color=COLOR[g], label=LABEL[g])
        ax.set_yticks(range(len(cats)), [c.replace("_", " ") for c in cats])
        ax.set_xlim(0, 1)
        ax.set_title(title)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_by_category.png"))
    plt.close(fig)

    p = pol[pol.group != "all"]
    names = list(dict.fromkeys(p.policy))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.55 * len(names) + 1.8), sharey=True)
    for ax, col, title in ((axes[0], "answered_unanswerable", "Answered an unanswerable question (lower is better)"),
                           (axes[1], "correct_answers_kept", "Correct answers kept (higher is better)")):
        for i, g in enumerate(GROUPS):
            v = p[p.group == g].set_index("policy").reindex(names)[col]
            ax.barh(np.arange(len(names)) + (i - 0.5) * 0.38, v, 0.38, color=COLOR[g], label=LABEL[g])
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=10)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_abstention_policies.png"))
    plt.close(fig)

    if len(cal):
        c = cal[cal.confidence == "max_prob"]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        x = np.arange(len(c))
        for i, g in enumerate(GROUPS):
            ax.bar(x + (i - 0.5) * 0.38, c[f"ece_{g}"], 0.38, color=COLOR[g], label=LABEL[g])
        ax.set_xticks(x, c.calibration, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("ECE (lower is better)")
        ax.set_title("Calibration error before and after post-hoc calibration (max_prob)")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(out, "fig4_calibration.png"))
        plt.close(fig)


# ----------------------------------------------------------------------------- report
def _md(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return "(not available)"
    return df.round(3).to_markdown(index=False)


def write_results_md(out, meta, summ, gaps, pol, pgaps, cal, var, cats):
    cols = ["group", "questions", "accuracy_answerable", "false_abstention", "missed_abstention_unanswerable",
            "own_doc_retrieved", "ece_max_prob", "overconfidence_max_prob", "aurc_max_prob"]
    lines = [
        "# Results", "",
        f"Generator: `{meta.get('gen_model', '?')}`, prompt: `{meta.get('prompt_style', 'default')}`, "
        f"top-{meta.get('top_k', '?')} retrieval, "
        f"{int(summ.loc[summ.group == 'all', 'questions'].iloc[0])} main questions.",
        "", "## Overall", "", _md(summ[cols]), "",
        "## Indian minus non-Indian (bootstrap 95% CI; Holm-adjusted p across the family)", "", _md(gaps), "",
        "## Abstention policies (thresholds by 5-fold cross-validation)", "", _md(pol), "",
        "## Group gaps under each policy", "", _md(pgaps), "",
        "## Post-hoc calibration (5-fold cross-validation, answered questions)", "", _md(cal), "",
        "## Spelling variants (question uses another spelling or older name)", "", _md(var), "",
        "## By category", "", _md(cats), "",
        "Figures: fig1_reliability_risk_coverage.png, fig2_by_category.png, fig3_abstention_policies.png, "
        "fig4_calibration.png", "",
    ]
    with open(os.path.join(out, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines))


def main(pred_path: str = "results/predictions.json", out: str = "results", tau: float = 0.55):
    full, meta = load(pred_path)
    df = full[~full.variant].reset_index(drop=True)
    os.makedirs(out, exist_ok=True)
    summ = summary_table(df)
    gaps = gap_table(df)
    cats = category_table(df)
    thr = threshold_table(df)
    pol, masks = policies(df)
    pgaps = policy_gaps(df, masks)
    cal = calibration(df)
    var = variant_table(full)
    for name, t in (("summary", summ), ("group_gaps", gaps), ("by_category", cats), ("threshold_sweep", thr),
                    ("policies", pol), ("policy_gaps", pgaps), ("calibration", cal), ("variants", var)):
        t.to_csv(os.path.join(out, f"{name}.csv"), index=False)
    figures(df, pol, cal, out)
    write_results_md(out, meta, summ, gaps, pol, pgaps, cal, var, cats)
    report = {"meta": meta, "summary": summ.round(4).to_dict(orient="records"),
              "indian_minus_non_indian": gaps.round(4).to_dict(orient="records"),
              "policies": pol.round(4).to_dict(orient="records"),
              "calibration": cal.round(4).to_dict(orient="records"),
              "variants": var.round(4).to_dict(orient="records")}
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n===== SUMMARY =====")
    cols = ["group", "questions", "accuracy_answerable", "missed_abstention_unanswerable", "own_doc_retrieved",
            "ece_max_prob", "aurc_max_prob"]
    print(summ[cols].round(3).to_string(index=False))
    print("\n===== INDIAN MINUS NON-INDIAN (95% bootstrap CI, Holm-adjusted p) =====")
    print(gaps[["metric", "indian_minus_non_indian", "ci_low", "ci_high", "p_holm"]].round(3).to_string(index=False))
    print("\n===== ABSTENTION POLICIES (5-fold CV) =====")
    print(pol.round(3).to_string(index=False))
    if len(cal):
        print("\n===== CALIBRATION (ECE) =====")
        print(cal.round(3).to_string(index=False))
    if len(var):
        print("\n===== SPELLING VARIANTS =====")
        print(var.round(3).to_string(index=False))
    print(f"\nAll tables, figures and RESULTS.md written to {out}/")
    return report


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(args[:2] if args else []))
