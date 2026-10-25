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
