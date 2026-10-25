"""Phase 0 check: run `python scripts/check_env.py`. Every line should say OK."""
import importlib
import sys

print(f"Python {sys.version.split()[0]}")
if sys.version_info[:2] not in [(3, 11), (3, 12)]:
    print("WARNING: use Python 3.11 or 3.12 (3.13/3.14 break shap/numba and others).")

modules = [
    "torch", "torchvision", "numpy", "pandas", "sklearn", "PIL",
    "tqdm", "pytorch_grad_cam", "shap", "matplotlib", "streamlit",
]
failed = []
for name in modules:
    try:
        m = importlib.import_module(name)
        print(f"OK   {name:18s} {getattr(m, '__version__', '')}")
    except Exception as e:
        failed.append(name)
        print(f"FAIL {name:18s} {type(e).__name__}: {e}")

try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}  (False is fine locally; Colab T4 will be True)")
except Exception:
    pass

sys.exit(1 if failed else 0)
