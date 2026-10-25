# Results log

## Phase 3: imbalance x augmentation grid (MobileNetV3-Small, 12 epochs, seed 42, T4)
Ranked by VALIDATION macro-F1 (selection rule fixed in advance). Test set: 3,866 tiles (266 Deforestation).

| run | aug | imbalance | val macro-F1 | test acc | test macro-F1 | Defo precision | Defo recall | Defo F1 | Kaggle acc | Planet acc |
|---|---|---|---|---|---|---|---|---|---|---|
| **B_aug** | yes | none | **0.891** | 0.950 | **0.883** | 0.696 | 0.707 | **0.701** | 0.986 | 0.891 |
| baseline (A) | no | none | 0.880 | 0.943 | 0.868 | 0.640 | 0.688 | 0.663 | 0.987 | 0.872 |
| F_sampler_aug | yes | sampler | 0.878 | 0.936 | 0.864 | 0.570 | 0.778 | 0.658 | 0.987 | 0.853 |
| E_sampler | no | sampler | 0.876 | 0.946 | 0.865 | 0.700 | 0.613 | 0.653 | 0.985 | 0.882 |
| C_wloss | no | weighted loss | 0.872 | 0.937 | 0.859 | 0.612 | 0.688 | 0.648 | 0.981 | 0.864 |
| D_wloss_aug | yes | weighted loss | 0.862 | 0.928 | 0.860 | 0.529 | 0.861 | 0.655 | 0.984 | 0.836 |

Observations
- Augmentation is the only change that improved F1; it also cut overfitting (final train loss 0.078 vs 0.012, val loss ~0.12 vs ~0.21).
- Weighted loss / weighted sampler shift the Deforestation precision-recall trade-off (D: recall 0.86, precision 0.53; E: precision 0.70, recall 0.61) but do not improve F1.
- Wildfire is near-solved (F1 ~0.985). Kaggle-tile accuracy (~98.6%) is much higher than Planet (~87-89%); the Kaggle set has a known spatial shortcut (see FINDINGS_FROM_FIRST_ATTEMPT.md), so Planet accuracy is the honest difficulty measure.
- Caveats: single seed; val macro-F1 fluctuates ~0.02 between epochs; only 266 test Deforestation tiles.
- Selected configuration for Phases 5-6: **B_aug** (`--aug`, no imbalance handling).
