# Plan and scope rules

## Rules (read before every session)
1. **One phase at a time.** Do not start phase N+1 until phase N's "Done when" is true.
2. **Scope freeze.** No new datasets, formats, or models. New ideas go in `docs/FUTURE_WORK.md`.
3. **Always runnable.** Every phase ends with a commit and a working state.
4. **Data never goes in git.** Big files live on Google Drive / local `data/`.
5. **Notebooks hold no logic.** Logic lives in `src/` so it works the same locally and on Colab.

## Fixed decisions
- Framework: PyTorch (no TensorFlow).
- Python: 3.11 or 3.12 (not 3.13/3.14).
- Model: MobileNetV3-Small first; ResNet18 only as an optional second architecture.
- Classes (Option A): Normal / Wildfire / Deforestation. "Normal" is drawn from BOTH
  Kaggle `nowildfire` tiles and Planet tiles with no deforestation label, so the model
  cannot predict the class from the data source alone. Report accuracy per source.
- Datasets: Kaggle Wildfire Prediction (JPG) + Planet Amazon (JPG). NO Copernicus/GeoTIFF, NO Rasterio.
- Training on Colab T4 from a preprocessed zip on Drive; local machine is for code and CPU smoke tests.

## Phases
| # | Phase | Done when |
|---|---|---|
| 0 | Setup | `python scripts/check_env.py` all OK; repo on GitHub |
| 1 | Data + 3-class dataset | `labels.csv` exists; count table shows 3 classes and Normal from both sources |
| 2 | Baseline training | checkpoint + confusion matrix saved |
| 3 | Imbalance + augmentation grid | one results table (2x3 grid) + best model chosen |
| 4 | Shortcut / per-source check | per-source table + limitations paragraph |
| 5 | Explainability | image / Grad-CAM / SHAP figure per class |
| 6 | Dashboard | upload patch -> class + confidence + heatmap on CPU |
| 7 | Report, slides, demo | report, slides, 2-min backup demo video |

## Cut order if time runs short
1. Folium map  2. ResNet18  3. SHAP in dashboard  4. Sampler experiments
