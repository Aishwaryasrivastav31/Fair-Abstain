"""Compare several runs (e.g. flan-t5-small, flan-t5-base, few-shot prompt).

    python src/compare.py results results_base results_fewshot

Writes comparison.csv and comparison.md in the first directory.
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd


def main(dirs: list[str]):
    rows = []
    for d in dirs:
        path = os.path.join(d, "report.json")
        if not os.path.exists(path):
            print(f"skipping {d}: no report.json")
            continue
        rep = json.load(open(path))
        meta = rep.get("meta", {})
        run = f"{meta.get('gen_model', d).split('/')[-1]} / {meta.get('prompt_style', 'default')}"
        summ = {r["group"]: r for r in rep["summary"]}
        pol = pd.DataFrame(rep.get("policies", []))
        for g in ("all", "indian", "non_indian"):
            s = summ[g]
            r = {"run": run, "group": g, "accuracy_answerable": s["accuracy_answerable"],
                 "missed_abstention_unanswerable": s["missed_abstention_unanswerable"],
                 "ece_max_prob": s["ece_max_prob"], "aurc_max_prob": s["aurc_max_prob"]}
            if len(pol):
                best = pol[pol.group == "all"].sort_values("utility", ascending=False).iloc[0]["policy"]
                b = pol[(pol.policy == best) & (pol.group == g)].iloc[0]
                r.update({"best_policy": best, "best_policy_utility": b["utility"],
                          "best_policy_answered_unanswerable": b["answered_unanswerable"],
                          "best_policy_correct_kept": b["correct_answers_kept"]})
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(dirs[0], "comparison.csv"), index=False)
    with open(os.path.join(dirs[0], "comparison.md"), "w") as f:
        f.write("# Comparison of runs\n\n" + out.round(3).to_markdown(index=False) + "\n")
    pd.set_option("display.width", 220)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:] or ["results"])
