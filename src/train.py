"""Train + evaluate one configuration. Run from the repo root:

  python -m src.train --data data\\processed --name baseline
  python -m src.train --data data\\processed_smoke --name smoke --epochs 2 --freeze-epochs 1 --workers 0

Imbalance / augmentation experiments (Phase 3) are just flags:
  --imbalance none|weighted_loss|sampler      --aug
Model selection uses VALIDATION macro-F1; the TEST set is evaluated once at the end.
"""
import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from tqdm import tqdm

from src.data import make_loaders
from src.model import MODELS, build_model, set_trainable


def run_epoch(model, loader, device, criterion, optimizer=None, desc=""):
    train = optimizer is not None
    model.train(train)
    total, n, ys, ps = 0.0, 0, [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
            ys.append(y.cpu())
            ps.append(out.argmax(1).cpu())
    return total / n, torch.cat(ys).numpy(), torch.cat(ps).numpy()


def summarize(y, p, n_classes):
    labels = list(range(n_classes))
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=labels, zero_division=0)
    return dict(acc=float(accuracy_score(y, p)),
                macro_f1=float(f1_score(y, p, labels=labels, average="macro", zero_division=0)),
                per_class_f1=f1.tolist(), per_class_precision=pr.tolist(), per_class_recall=rc.tolist())


def save_confusion(cm, classes, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(len(classes)), classes, rotation=20)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test confusion matrix (counts, row %)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--model", default="mobilenet_v3_small", choices=MODELS)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--freeze-epochs", type=int, default=2, help="epochs with the backbone frozen (head only)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="head lr; backbone phase uses lr/10")
    ap.add_argument("--aug", action="store_true", help="flip/rot90/color-jitter augmentation")
    ap.add_argument("--imbalance", default="none", choices=["none", "weighted_loss", "sampler"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (for offline tests only)")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(a.out, a.name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(run_dir, "args.json"), "w"), indent=2)

    loaders, parts, classes, counts = make_loaders(a.data, a.batch_size, a.aug, a.imbalance, a.workers, a.seed)
    C = len(classes)
    print(f"device={device}  model={a.model}  aug={a.aug}  imbalance={a.imbalance}")
    print("train class counts:", dict(zip(classes, counts.tolist())))

    model = build_model(a.model, C, pretrained=not a.no_pretrained).to(device)
    weight = None
    if a.imbalance == "weighted_loss":
        weight = torch.tensor(counts.sum() / (C * counts), dtype=torch.float32, device=device)
        print("loss weights:", dict(zip(classes, [round(float(w), 2) for w in weight])))
    criterion = nn.CrossEntropyLoss(weight=weight)

    history, best, cur_frozen, opt = [], -1.0, None, None
    ckpt = os.path.join(run_dir, "best.pt")
    for epoch in range(1, a.epochs + 1):
        frozen = epoch <= a.freeze_epochs
        if opt is None or frozen != cur_frozen:  # (re)build optimizer when the freeze state changes
            set_trainable(model, backbone=not frozen)
            lr = a.lr if frozen else a.lr / 10
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
            cur_frozen = frozen
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, loaders["train"], device, criterion, opt, f"train {epoch}")
        va_loss, vy, vp = run_epoch(model, loaders["val"], device, criterion, None, f"val {epoch}")
        s = summarize(vy, vp, C)
        row = dict(epoch=epoch, frozen=frozen, train_loss=tr_loss, val_loss=va_loss, val_acc=s["acc"],
                   val_macro_f1=s["macro_f1"], **{f"val_f1_{c}": f for c, f in zip(classes, s["per_class_f1"])})
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
        star = ""
        if s["macro_f1"] > best:
            best = s["macro_f1"]
            torch.save(dict(model=a.model, classes=classes, state_dict=model.state_dict(), args=vars(a)), ckpt)
            star = "  <- best"
        f1s = " ".join(f"{c[:4]}={f:.3f}" for c, f in zip(classes, s["per_class_f1"]))
        print(f"epoch {epoch:2d}/{a.epochs} {'[head]' if frozen else '[full]'} "
              f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={s['acc']:.4f} "
              f"val_macroF1={s['macro_f1']:.4f} ({f1s}) {time.time()-t0:.0f}s{star}")

    # ---- final test evaluation with the best (by val macro-F1) checkpoint ----
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    _, ty, tp = run_epoch(model, loaders["test"], device, criterion, None, "test")
    res = summarize(ty, tp, C)
    cm = confusion_matrix(ty, tp, labels=list(range(C)))
    test_df = parts["test"].assign(pred=tp)
    test_df["correct"] = test_df["pred"] == test_df["y"]
    per_source = {s: dict(n=int(len(g)), acc=float(g["correct"].mean())) for s, g in test_df.groupby("source")}
    per_label_source = {f"{l}|{s}": dict(n=int(len(g)), recall=float(g["correct"].mean()))
                        for (l, s), g in test_df.groupby(["label", "source"])}
    out = dict(name=a.name, config=vars(a), best_val_macro_f1=best, test=res, classes=classes,
               confusion_matrix=cm.tolist(), per_source=per_source, per_label_source=per_label_source)
    json.dump(out, open(os.path.join(run_dir, "metrics.json"), "w"), indent=2)
    save_confusion(cm, classes, os.path.join(run_dir, "confusion_matrix.png"))
    test_df[["path", "label", "source", "y", "pred"]].to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)

    print("\n=== TEST RESULTS ===")
    print(f"accuracy={res['acc']:.4f}  macro_F1={res['macro_f1']:.4f}")
    for c, p_, r_, f_ in zip(classes, res["per_class_precision"], res["per_class_recall"], res["per_class_f1"]):
        print(f"  {c:14s} precision={p_:.3f} recall={r_:.3f} F1={f_:.3f}")
    print("confusion matrix (rows=true, cols=pred):", classes)
    print(cm)
    print("per-source accuracy:", {k: round(v['acc'], 4) for k, v in per_source.items()})
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
