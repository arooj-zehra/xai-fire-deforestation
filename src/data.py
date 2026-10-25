"""Dataset + loaders for the 3-class tile dataset built by scripts/build_dataset.py."""
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
MEAN = (0.485, 0.456, 0.406)  # ImageNet statistics (the backbones were pretrained on ImageNet)
STD = (0.229, 0.224, 0.225)


class Rot90:
    """Random 0/90/180/270 degree rotation (satellite tiles have no 'up')."""

    def __call__(self, img):
        k = int(torch.randint(0, 4, (1,)))
        return img.rotate(90 * k) if k else img


def get_transforms(train, aug):
    ops = []
    if train and aug:
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), Rot90(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)]
    ops += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(ops)


def load_labels(root):
    df = pd.read_csv(os.path.join(root, "labels.csv"))
    with open(os.path.join(root, "meta.json")) as f:
        classes = json.load(f)["classes"]
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)})
    return df, classes


class TileDataset(Dataset):
    def __init__(self, root, df, transform):
        self.root, self.transform = root, transform
        self.paths = df["path"].tolist()
        self.y = df["y"].to_numpy()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.paths[i])).convert("RGB")
        return self.transform(img), int(self.y[i])


def make_loaders(root, batch_size=64, aug=False, imbalance="none", workers=2, seed=42):
    """imbalance: 'none' | 'weighted_loss' (handled in train.py via class_counts) | 'sampler'."""
    df, classes = load_labels(root)
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    counts = np.bincount(parts["train"]["y"], minlength=len(classes))
    pin = torch.cuda.is_available()
    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=pin, persistent_workers=workers > 0)

    train_ds = TileDataset(root, parts["train"], get_transforms(True, aug))
    if imbalance == "sampler":
        w = (1.0 / counts)[parts["train"]["y"].to_numpy()]
        g = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(train_ds), replacement=True, generator=g)
        train_loader = DataLoader(train_ds, sampler=sampler, **kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **kw)

    eval_tf = get_transforms(False, False)
    loaders = {
        "train": train_loader,
        "val": DataLoader(TileDataset(root, parts["val"], eval_tf), shuffle=False, **kw),
        "test": DataLoader(TileDataset(root, parts["test"], eval_tf), shuffle=False, **kw),
    }
    return loaders, parts, classes, counts
