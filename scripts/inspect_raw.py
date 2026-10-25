"""Phase 1 helper: show folder layout + file counts under data/raw, and the head of the Planet CSV."""
import os
import sys
from collections import Counter

root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
max_depth = 5

for dirpath, dirnames, filenames in os.walk(root):
    depth = dirpath[len(root):].count(os.sep)
    if depth > max_depth:
        dirnames[:] = []
        continue
    if filenames:
        exts = Counter(os.path.splitext(f)[1].lower() for f in filenames)
        print(f"{dirpath}  files={len(filenames)}  {dict(exts)}")
    else:
        print(f"{dirpath}  (dirs={len(dirnames)})")

for dirpath, _, filenames in os.walk(root):
    for f in filenames:
        if f.lower().endswith(".csv"):
            p = os.path.join(dirpath, f)
            print(f"\n--- {p} (first 5 lines) ---")
            with open(p, encoding="utf-8", errors="replace") as fh:
                for _ in range(5):
                    print(fh.readline().rstrip())
