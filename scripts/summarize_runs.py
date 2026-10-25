"""Collect every outputs/<run>/metrics.json into one results table.

Usage: python scripts/summarize_runs.py --outputs <path to outputs folder>
Runs are ranked by VALIDATION macro-F1 (never by test), so the choice of best model does not peek at test data.
"""
import argparse
import glob
import json
import os

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--outputs", default="outputs")
a = ap.parse_args()

rows = []
for p in sorted(glob.glob(os.path.join(a.outputs, "*", "metrics.json"))):
    m = json.load(open(p))
    c, t, ps = m["config"], m["test"], m["per_source"]
    d = m["classes"].index("Deforestation")
    rows.append(dict(
        run=m["name"], model=c["model"], aug=c["aug"], imbalance=c["imbalance"], epochs=c["epochs"],
        val_macro_f1=m["best_val_macro_f1"],
        test_acc=t["acc"], test_macro_f1=t["macro_f1"],
        defo_precision=t["per_class_precision"][d], defo_recall=t["per_class_recall"][d], defo_f1=t["per_class_f1"][d],
        kaggle_acc=ps.get("kaggle", {}).get("acc"), planet_acc=ps.get("planet", {}).get("acc"),
    ))
if not rows:
    raise SystemExit(f"no metrics.json found under {a.outputs}")

df = pd.DataFrame(rows).sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(df.round(3).to_string(index=False))
out = os.path.join(a.outputs, "results.csv")
df.to_csv(out, index=False)
print(f"\nBest by VALIDATION macro-F1: {df.loc[0, 'run']}")
print(f"Saved {out}")
