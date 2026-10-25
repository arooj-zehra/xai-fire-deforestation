"""Phase 4: does the model do more than read the data source or crude colour statistics?

Run (Colab or local, from the repo root):
  python scripts/source_report.py --data /content/data/processed --run <outputs>/B_aug

Needs <run>/test_predictions.csv (written by src.train) and the processed dataset (labels.csv + images).
Compares the CNN against two trivial baselines on the SAME test tiles:
  * source-only : predict the most common class of the tile's data source (kaggle / planet)
  * colour-stats: gradient boosting on 8 numbers per image (mean/std per RGB channel, 10th/90th pct brightness)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--run", required=True)
a = ap.parse_args()

classes = json.load(open(os.path.join(a.data, "meta.json")))["classes"]
cidx = {c: i for i, c in enumerate(classes)}
lab = pd.read_csv(os.path.join(a.data, "labels.csv"))
lab["y"] = lab["label"].map(cidx)
train, test = lab[lab.split == "train"].reset_index(drop=True), lab[lab.split == "test"].reset_index(drop=True)

cnn = pd.read_csv(os.path.join(a.run, "test_predictions.csv"))
test = test.merge(cnn[["path", "pred"]], on="path", how="left")
assert test["pred"].notna().all(), "test_predictions.csv does not match this dataset"
test["pred"] = test["pred"].astype(int)


def feats(path):
    x = np.asarray(Image.open(os.path.join(a.data, path)).convert("RGB"), dtype=np.float32)
    g = x.mean(2)
    return np.concatenate([x.mean((0, 1)), x.std((0, 1)), np.percentile(g, [10, 90])])


def X(df, desc):
    return np.stack([feats(p) for p in tqdm(df["path"], desc=desc)])


# baselines
src_major = train.groupby("source")["y"].agg(lambda s: int(s.value_counts().idxmax()))
test["pred_source_only"] = test["source"].map(src_major).astype(int)
gb = HistGradientBoostingClassifier(random_state=42).fit(X(train, "features train"), train["y"])
test["pred_colour"] = gb.predict(X(test, "features test"))


def stats(df, col):
    f = f1_score(df["y"], df[col], labels=list(range(len(classes))), average=None, zero_division=0)
    return dict(acc=accuracy_score(df["y"], df[col]),
                macro_f1=f1_score(df["y"], df[col], labels=list(range(len(classes))), average="macro", zero_division=0),
                **{f"f1_{c}": f[i] for i, c in enumerate(classes)})


def within(df, col):
    """macro-F1 inside one source, over the classes that exist there."""
    labs = sorted(df["y"].unique())
    return f1_score(df["y"], df[col], labels=labs, average="macro", zero_division=0)


rows = {"CNN (this run)": "pred", "colour-stats GB": "pred_colour", "source-only": "pred_source_only"}
tab = pd.DataFrame({k: stats(test, c) for k, c in rows.items()}).T
for k, c in rows.items():
    for s in ("kaggle", "planet"):
        tab.loc[k, f"{s}_within_macroF1"] = within(test[test.source == s], c)
pd.set_option("display.width", 220)
print(f"\n=== Test tiles: {len(test)} ===")
print(tab.round(3).to_string())

print("\n=== CNN recall per (true class, source) ===")
g = test.assign(ok=test.y == test.pred).groupby(["label", "source"])["ok"].agg(["size", "mean"])
print(g.rename(columns={"size": "n", "mean": "recall"}).round(3).to_string())

print("\n=== Where the CNN's Planet errors go (rows=true, cols=pred), Planet tiles only ===")
pl = test[test.source == "planet"]
print(pd.crosstab(pl["label"], pl["pred"].map(dict(enumerate(classes))), margins=True).to_string())

out = dict(table=tab.round(4).to_dict(orient="index"),
           cnn_recall_by_label_source={f"{l}|{s}": dict(n=int(r["size"]), recall=float(r["mean"])) for (l, s), r in g.iterrows()})
json.dump(out, open(os.path.join(a.run, "source_report.json"), "w"), indent=2)
print(f"\nSaved {os.path.join(a.run, 'source_report.json')}")
