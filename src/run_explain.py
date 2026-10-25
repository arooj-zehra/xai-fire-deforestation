"""Phase 5: Grad-CAM + SHAP figures and a faithfulness check for a trained run.

  python -m src.run_explain --data /content/data/processed --run <outputs>/B_aug

Writes <run>/explain/: explanations.png, shap_channels.png, faithfulness.json, selected_tiles.csv
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import TileDataset, get_transforms, load_labels
from src.explain import (cam_overlay, denorm, gradcam, load_checkpoint, predict_proba, shap_maps, shap_overlay)

GRID = 7  # Grad-CAM native resolution for 224px input (MobileNetV3 / ResNet18)


def batches(n, size):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--per-class", type=int, default=20, help="correct tiles per class used for statistics")
    ap.add_argument("--n-bg", type=int, default=50, help="SHAP background tiles")
    ap.add_argument("--nsamples", type=int, default=50, help="SHAP expected-gradients samples per tile")
    ap.add_argument("--cells", type=int, default=10, help="of 49 Grad-CAM cells masked in the faithfulness test")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_checkpoint(os.path.join(a.run, "best.pt"), device)
    out_dir = os.path.join(a.run, "explain")
    os.makedirs(out_dir, exist_ok=True)
    C = len(classes)

    df, _ = load_labels(a.data)
    preds = pd.read_csv(os.path.join(a.run, "test_predictions.csv"))[["path", "pred"]]
    test = df[df["split"] == "test"].merge(preds, on="path").reset_index(drop=True)

    # ---- choose tiles: per class, `per_class` correct ones (stats) + 1 error; figure uses 2 correct + 1 error ----
    sel = []
    for c in range(C):
        ok = test[(test.y == c) & (test.pred == c)]
        bad = test[(test.y == c) & (test.pred != c)]
        ok = ok.sample(min(a.per_class, len(ok)), random_state=a.seed)
        for k, (_, r) in enumerate(ok.iterrows()):
            sel.append({**r.to_dict(), "kind": "correct", "fig": k < 2})
        if len(bad):
            r = bad.sample(1, random_state=a.seed).iloc[0]
            sel.append({**r.to_dict(), "kind": "error", "fig": True})
    sel = pd.DataFrame(sel).reset_index(drop=True)
    sel.drop(columns=["y"]).assign(true=sel["label"]).to_csv(os.path.join(out_dir, "selected_tiles.csv"), index=False)

    ds = TileDataset(a.data, sel.assign(y=sel["y"]), get_transforms(False, False))
    X = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    probs = predict_proba(model, X).cpu().numpy()
    pred = probs.argmax(1)
    assert (pred == sel["pred"].to_numpy()).all(), "model predictions differ from test_predictions.csv (wrong checkpoint?)"
    print(f"{len(sel)} tiles selected; device={device}")

    # ---- Grad-CAM (target = predicted class) ----
    cams = np.concatenate([gradcam(model, X[s], pred[s]) for s in batches(len(X), 32)])

    # ---- SHAP (expected gradients, target = predicted class) ----
    train = df[df["split"] == "train"].sample(a.n_bg, random_state=a.seed)
    bg_ds = TileDataset(a.data, train, get_transforms(False, False))
    bg = torch.stack([bg_ds[i][0] for i in range(len(bg_ds))]).to(device)
    smaps = np.concatenate([shap_maps(model, X[s], pred[s], bg, a.nsamples) for s in batches(len(X), 8)])
    print("SHAP done", smaps.shape)

    # ---- figure: image | Grad-CAM | SHAP ----
    fig_rows = sel.index[sel["fig"]].tolist()
    fig, ax = plt.subplots(len(fig_rows), 3, figsize=(7.2, 2.35 * len(fig_rows)))
    for r, i in enumerate(fig_rows):
        rgb = denorm(X[i])
        ax[r, 0].imshow(rgb)
        ax[r, 1].imshow(cam_overlay(rgb, cams[i]))
        ax[r, 2].imshow(shap_overlay(rgb, smaps[i]))
        ax[r, 0].set_title(f"true {sel.label[i]} | pred {classes[pred[i]]} {probs[i, pred[i]]:.2f}"
                           f"{'' if sel.kind[i] == 'correct' else '  (ERROR)'}", fontsize=7, loc="left")
        for k in range(3):
            ax[r, k].axis("off")
    for k, t in enumerate(["tile", "Grad-CAM (red = important)", "SHAP (red = for, blue = against)"]):
        ax[0, k].text(0.5, 1.28, t, transform=ax[0, k].transAxes, ha="center", fontsize=8, weight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "explanations.png"), dpi=90)
    plt.close(fig)

    # ---- per-channel SHAP importance (correct tiles only) ----
    correct = (sel.kind == "correct").to_numpy()
    ch = {}
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    w = 0.25
    for c in range(C):
        idx = np.where(correct & (pred == c))[0]
        if len(idx) == 0:
            continue
        share = np.abs(smaps[idx]).sum((2, 3)).mean(0)
        share = share / share.sum()
        ch[classes[c]] = dict(zip("RGB", share.round(4).tolist()))
        ax.bar(np.arange(3) + (c - 1) * w, share, w, label=classes[c], color=["#4c78a8", "#e45756", "#54a24b"][c])
    ax.set_xticks(range(3), ["Red", "Green", "Blue"])
    ax.set_ylabel("share of mean |SHAP|")
    ax.set_title("SHAP importance per colour channel (correct tiles)", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "shap_channels.png"), dpi=110)
    plt.close(fig)
    print("per-channel SHAP share:", json.dumps(ch))

    # ---- faithfulness: does hiding Grad-CAM's top cells hurt the prediction more than hiding random cells? ----
    k = a.cells
    idx = np.where(correct)[0]
    drops_cam, drops_rand = [], []
    Xc = X[idx].clone()
    cell = Xc.shape[-1] // GRID
    for j, i in enumerate(idx):
        score = cams[i].reshape(GRID, cell, GRID, cell).mean((1, 3)).ravel()
        for kind, cells in (("cam", np.argsort(-score)[:k]), ("rand", rng.choice(GRID * GRID, k, replace=False))):
            m = torch.ones(GRID * GRID)
            m[torch.as_tensor(cells)] = 0
            m = m.view(GRID, 1, GRID, 1).expand(GRID, cell, GRID, cell).reshape(GRID * cell, GRID * cell).to(device)
            xm = (X[i] * m).unsqueeze(0)  # 0 == dataset-mean colour after normalisation
            p = predict_proba(model, xm)[0, pred[i]].item()
            (drops_cam if kind == "cam" else drops_rand).append(probs[i, pred[i]] - p)
    drops_cam, drops_rand = np.array(drops_cam), np.array(drops_rand)
    res = dict(n_tiles=int(len(idx)), cells_masked=f"{k} of {GRID * GRID}",
               mean_confidence_drop_gradcam=float(drops_cam.mean()), mean_confidence_drop_random=float(drops_rand.mean()),
               gradcam_beats_random_fraction=float((drops_cam > drops_rand).mean()), per_channel_shap_share=ch)
    for c in range(C):
        m = pred[idx] == c
        if m.any():
            res[f"drop_gradcam_{classes[c]}"] = float(drops_cam[m].mean())
            res[f"drop_random_{classes[c]}"] = float(drops_rand[m].mean())
    json.dump(res, open(os.path.join(out_dir, "faithfulness.json"), "w"), indent=2)
    print("\n=== Faithfulness (confidence drop when masking the top-k Grad-CAM cells vs k random cells) ===")
    print(json.dumps({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in res.items() if kk != "per_channel_shap_share"}, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
