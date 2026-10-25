"""Phase 1: build the 3-class dataset (Normal / Wildfire / Deforestation).

Reads  data/raw/wildfire (Kaggle) and data/raw/planet (Planet Amazon)
Writes data/processed/images/*.jpg (224x224), labels.csv, meta.json

Option A: "Normal" is drawn from BOTH sources, so the data source alone cannot reveal the class.

Usage (from repo root, venv active):
    python scripts\\build_dataset.py --cap 200     # quick smoke test (small)
    python scripts\\build_dataset.py               # full build (cap 8000 per class/source cell)
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True  # some Kaggle JPEGs are truncated

CLASSES = ["Normal", "Wildfire", "Deforestation"]  # fixed order = class index 0,1,2

# Planet tags that mean forest loss / clearing. NOTE: "clear" is a WEATHER tag (clear sky), not clearing.
DEFOREST_TAGS = {"slash_burn", "selective_logging", "conventional_mine", "artisinal_mine", "bare_ground"}
EXCLUDE_TAGS = {"cloudy"}  # can't see the ground
THUMB = 24  # near-duplicate check uses 24x24 RGB thumbnails
MAD_MAX = 0.3  # only near-pixel-identical tiles; distinct canopy tiles are >= 0.6 apart (see docs/FINDINGS_FROM_FIRST_ATTEMPT.md)
SEED = 42


def process(job):
    """Resize one image to size x size and save it. Returns (ok, thumb_or_error)."""
    src, dst, size, want_thumb = job
    try:
        with Image.open(src) as im:
            im = im.convert("RGB").resize((size, size), Image.LANCZOS)
            im.save(dst, "JPEG", quality=95)
            thumb = None
            if want_thumb:
                t = im.resize((THUMB, THUMB), Image.BOX)
                thumb = np.asarray(t, dtype=np.float32).ravel()
        return True, thumb
    except Exception as e:  # never let one bad file kill the run
        return False, f"{type(e).__name__}: {e}"


def collect_kaggle(raw):
    root = os.path.join(raw, "wildfire")
    rows = []
    for split_dir in ("train", "valid", "test"):  # we ignore Kaggle's split and re-split ourselves
        for folder, cls in (("wildfire", "Wildfire"), ("nowildfire", "Normal")):
            for p in sorted(glob.glob(os.path.join(root, split_dir, folder, "*.jpg"))):
                rows.append(dict(orig_path=p, label=cls, source="kaggle", tags=""))
    if not rows:
        sys.exit(f"No Kaggle images found under {root}")
    return rows


def collect_planet(raw):
    csvs = glob.glob(os.path.join(raw, "planet", "**", "train_classes.csv"), recursive=True)
    if not csvs:
        sys.exit("train_classes.csv not found under data/raw/planet")
    img_dir = os.path.join(os.path.dirname(csvs[0]), "train-jpg")
    df = pd.read_csv(csvs[0])
    rows, tag_counts, skipped_cloudy = [], Counter(), 0
    for name, tags in zip(df["image_name"], df["tags"]):
        t = set(str(tags).split())
        if t & EXCLUDE_TAGS:
            skipped_cloudy += 1
            continue
        hit = t & DEFOREST_TAGS
        if hit:
            label = "Deforestation"
            tag_counts.update(hit)
        else:
            label = "Normal"
        rows.append(dict(orig_path=os.path.join(img_dir, name + ".jpg"), label=label, source="planet", tags=tags))
    print(f"Planet: {len(df)} rows, skipped {skipped_cloudy} cloudy.")
    print("Deforestation tag counts (a tile can have several):", dict(tag_counts))
    return rows


def cap_cells(df, cap):
    parts = []
    for (label, source), g in df.groupby(["label", "source"]):
        parts.append(g.sample(n=min(cap, len(g)), random_state=SEED))
    return pd.concat(parts, ignore_index=True)


def dedup_planet(df, thumbs, mad_max):
    """Drop near-duplicate Planet tiles. Earlier rows win, so Deforestation (placed first) is kept."""
    idx = np.where(df["source"].to_numpy() == "planet")[0]
    T = np.stack([thumbs[i] for i in idx]).astype(np.float32)
    dropped = set()
    chunk = 500
    for s in tqdm(range(0, len(idx), chunk), desc="dedup"):
        D = cdist(T[s:s + chunk], T, metric="cityblock") / T.shape[1]
        for r in range(D.shape[0]):
            i = s + r
            if i in dropped:
                continue
            for j in np.where(D[r] <= mad_max)[0]:
                if j > i:
                    dropped.add(int(j))
    return [int(idx[j]) for j in dropped]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--cap", type=int, default=8000, help="max images per (class, source) cell")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 2))
    ap.add_argument("--mad-max", type=float, default=MAD_MAX, help="near-duplicate threshold (see scripts/dedup_diagnostic.py)")
    ap.add_argument("--max-dup-frac", type=float, default=0.02,
                    help="abort (without deleting anything) if dedup would drop more than this fraction of Planet tiles")
    a = ap.parse_args()

    df = pd.DataFrame(collect_kaggle(a.raw) + collect_planet(a.raw))
    print("\nAvailable before capping:")
    print(pd.crosstab(df["label"], df["source"]))

    df = cap_cells(df, a.cap)
    # Planet Deforestation first, so dedup keeps those over any look-alike Normal tile
    df["_o"] = df["label"].map({"Deforestation": 0, "Wildfire": 1, "Normal": 2})
    df = df.sort_values(["_o", "source", "orig_path"]).drop(columns="_o").reset_index(drop=True)
    df["file"] = [f"{s}_{i:06d}.jpg" for i, s in enumerate(df["source"])]

    img_out = os.path.join(a.out, "images")
    os.makedirs(img_out, exist_ok=True)
    jobs = [(r.orig_path, os.path.join(img_out, r.file), a.size, r.source == "planet") for r in df.itertuples()]

    thumbs, bad = {}, []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (ok, res) in enumerate(tqdm(ex.map(process, jobs, chunksize=32), total=len(jobs), desc="resize")):
            if ok:
                if res is not None:
                    thumbs[i] = res
            else:
                bad.append((i, res))
    if bad:
        print(f"\n{len(bad)} unreadable images skipped, e.g. {bad[:2]}")
    df = df.drop(index=[i for i, _ in bad])
    thumbs = {k: v for k, v in thumbs.items() if k in set(df.index)}

    drop = dedup_planet(df, thumbs, a.mad_max)
    n_planet = int((df["source"] == "planet").sum())
    print(f"\nNear-duplicates found (Planet): {len(drop)} of {n_planet}  "
          f"(Deforestation among them: {int((df.loc[drop, 'label'] == 'Deforestation').sum())})")
    if len(drop) > a.max_dup_frac * n_planet:
        sys.exit(f"STOP: dedup would drop {len(drop)}/{n_planet} Planet tiles (> {a.max_dup_frac:.0%}). "
                 "That looks like over-pruning. Nothing was deleted or written; send this output to Claude.")
    for i in drop:
        try:
            os.remove(os.path.join(img_out, df.loc[i, "file"]))
        except OSError:
            pass
    df = df.drop(index=drop).reset_index(drop=True)

    # stratified 70/15/15 on (label, source)
    key = df["label"] + "|" + df["source"]
    small = key.value_counts()[lambda c: c < 10]
    if len(small):
        sys.exit(f"Too few images in some class/source cells to split: {small.to_dict()}")
    tr, tmp = train_test_split(df.index, test_size=0.30, stratify=key, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=key[tmp], random_state=SEED)
    df["split"] = ""
    df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"

    df["path"] = "images/" + df["file"]
    df["orig_path"] = df["orig_path"].str.replace("\\", "/", regex=False)
    df[["path", "label", "source", "split", "tags", "orig_path"]].to_csv(
        os.path.join(a.out, "labels.csv"), index=False)
    with open(os.path.join(a.out, "meta.json"), "w") as f:
        json.dump(dict(classes=CLASSES, size=a.size, cap=a.cap, seed=SEED,
                       deforest_tags=sorted(DEFOREST_TAGS), mad_max=a.mad_max), f, indent=2)

    print("\n=== FINAL COUNTS: class x source x split ===")
    print(pd.crosstab([df["label"], df["source"]], df["split"], margins=True))
    print("\nTotals by class:\n", df["label"].value_counts().reindex(CLASSES))

    normal_sources = set(df.loc[df["label"] == "Normal", "source"])
    assert normal_sources == {"kaggle", "planet"}, f"Normal must come from both sources, got {normal_sources}"
    for s in ("train", "val", "test"):
        assert set(df.loc[df["split"] == s, "label"]) == set(CLASSES), f"{s} is missing a class"
    print("\nOK: 3 classes in every split, Normal drawn from both sources.")


if __name__ == "__main__":
    main()
