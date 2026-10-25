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

## Phase 3 additions: repeat seeds and second architecture
| run | model | aug | seed | val macro-F1 | test macro-F1 | Defo F1 | Planet acc |
|---|---|---|---|---|---|---|---|
| baseline | MobileNetV3-S | no | 42 | 0.880 | 0.868 | 0.663 | 0.872 |
| baseline_s7 | MobileNetV3-S | no | 7 | 0.878 | 0.859 | 0.646 | 0.876 |
| B_aug | MobileNetV3-S | yes | 42 | 0.891 | 0.883 | 0.701 | 0.891 |
| B_aug_s7 | MobileNetV3-S | yes | 7 | 0.883 | 0.879 | 0.688 | 0.892 |
| G_resnet18_aug | ResNet18 | yes | 42 | 0.880 | 0.874 | 0.685 | 0.872 |

- Two-seed means: baseline 0.864 macro-F1 / 0.655 Defo F1; augmentation 0.881 / 0.695. Augmentation helps in both seeds (+~0.017 macro-F1, +~0.04 Defo F1); seed ranges do not overlap, but n=2.
- ResNet18 + aug (0.874) does not beat MobileNetV3-Small + aug and is ~2x slower (about 85 s vs 70 s per epoch on a T4) and much larger (ResNet18 has roughly 11M parameters, MobileNetV3-Small roughly 1.5M).
- FINAL MODEL: MobileNetV3-Small + augmentation (`B_aug`, seed 42 checkpoint).

## Phase 4: shortcut / per-source check (final model B_aug, 3,866 test tiles)
| predictor | acc | macro-F1 | Defo F1 | Kaggle within-source macro-F1 | Planet within-source macro-F1 |
|---|---|---|---|---|---|
| CNN (B_aug) | 0.950 | 0.883 | 0.701 | 0.986 | 0.817 |
| colour-stats gradient boosting (8 features) | 0.890 | 0.749 | 0.413 | 0.923 | 0.659 |
| source-only (majority class per source) | 0.621 | 0.429 | 0.000 | 0.333 | 0.450 |

CNN recall by (class, source): Deforestation|Planet 0.707 (n=266), Normal|Planet 0.932, Normal|Kaggle 0.989, Wildfire|Kaggle 0.983.
Planet errors: 78 Deforestation tiles called Normal, 82 Normal tiles called Deforestation (no Wildfire predictions on Planet tiles).

Report points
- The CNN uses image content beyond the data source or global colour: +0.13 macro-F1 over colour statistics overall, +0.16 within Planet.
- Kaggle Normal-vs-Wildfire is solvable to 92% with 8 colour numbers, consistent with the known spatial/visual shortcut in that dataset; Planet within-source F1 (0.82) is the honest difficulty measure.
- The model never confuses classes across sources, i.e. it effectively solves two binary tasks and can recognise the source. Option A ensures the label is not determined by source within each dataset.
