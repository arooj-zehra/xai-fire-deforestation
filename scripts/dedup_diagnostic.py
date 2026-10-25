"""Measure how close the nearest-neighbour Planet tiles really are, so we can pick a safe duplicate threshold.

Usage: python scripts\\dedup_diagnostic.py
Prints the distribution of nearest-neighbour distances and saves outputs/dedup_pairs.png
(example pairs at increasing distance) for eyeballing.
"""
import glob
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy.spatial.distance import cdist
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
N, THUMB, SIZE = 4000, 24, 224

csv = glob.glob("data/raw/planet/**/train_classes.csv", recursive=True)
if not csv:
    sys.exit("run from the repo root; train_classes.csv not found")
img_dir = os.path.join(os.path.dirname(csv[0]), "train-jpg")
df = pd.read_csv(csv[0])
df = df[~df["tags"].str.contains("cloudy")]
names = df["image_name"].tolist()
random.Random(0).shuffle(names)
names = names[:N]

thumbs = []
for n in tqdm(names, desc="thumbs"):
    im = Image.open(os.path.join(img_dir, n + ".jpg")).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    thumbs.append(np.asarray(im.resize((THUMB, THUMB), Image.BOX), dtype=np.float32).ravel())
T = np.stack(thumbs)

D = cdist(T, T, metric="cityblock") / T.shape[1]
np.fill_diagonal(D, np.inf)
nn_idx = D.argmin(axis=1)
nn = D.min(axis=1)

print(f"\nNearest-neighbour distance among {len(names)} random Planet tiles (per-value, 0-255 scale)")
for q in (0, 0.1, 1, 5, 25, 50):
    print(f"  {q:>5}th percentile: {np.percentile(nn, q):.2f}")
for t in (0.5, 1, 2, 3, 5, 8):
    print(f"  tiles with a neighbour within {t:>3}: {(nn <= t).sum()}")

os.makedirs("outputs", exist_ok=True)
targets = [float(nn.min()), 1.0, 2.0, 3.0, 5.0, 8.0]
rows = []
for t in targets:
    i = int(np.argmin(np.abs(nn - t)))
    rows.append((i, int(nn_idx[i]), nn[i]))
fig, ax = plt.subplots(len(rows), 2, figsize=(5, 2.6 * len(rows)))
for r, (i, j, d) in enumerate(rows):
    for c, k in enumerate((i, j)):
        ax[r, c].imshow(Image.open(os.path.join(img_dir, names[k] + ".jpg")))
        ax[r, c].axis("off")
    ax[r, 0].set_title(f"distance {d:.2f}", fontsize=9, loc="left")
plt.tight_layout()
plt.savefig("outputs/dedup_pairs.png", dpi=80)
print("\nSaved outputs/dedup_pairs.png  (attach it in the chat)")
