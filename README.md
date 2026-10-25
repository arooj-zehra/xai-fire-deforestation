# Explainable Wildfire & Deforestation Detection (CNN + Grad-CAM + SHAP)

Programming for AI course project (solo).

3-class satellite tile classifier — **Normal / Wildfire / Deforestation** — with
MobileNetV3-Small (PyTorch), Grad-CAM and SHAP explanations, and a Streamlit dashboard.

## Quick start (Windows)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
```

## Layout

| Folder | Purpose |
|---|---|
| `src/` | Library code (dataset, model, train, explain) |
| `scripts/` | One-off command-line scripts (env check, data prep) |
| `notebooks/` | Colab setup notebook only; no logic lives in notebooks |
| `app/` | Streamlit dashboard |
| `data/` | Local data (git-ignored) |
| `outputs/` | Checkpoints, metrics, figures (git-ignored) |
| `docs/` | Plan, scope rules, decisions |

Read `docs/PLAN.md` first. It has the phases and the scope rules.

## Dashboard (Phase 6)

1. Download `best.pt` from Drive (`xai-fire-deforestation/outputs/B_aug/`) to `outputs\B_aug\best.pt`.
2. Run from the repo root with the venv active:

```
streamlit run app/app.py
```

Upload a 224x224-ish satellite tile (or use "try a random test tile" if `data\processed` exists).
SHAP is optional and slow on CPU (needs `data\processed` for background tiles).

## Results

See `docs/RESULTS.md` for the experiment grid, the shortcut check, and the explainability findings.
