# Findings carried over from the first attempt (use in the report)

- Kaggle Wildfire has a spatial shortcut: a k-NN on (lat, lon) alone gets ~99.0-100% accuracy
  (random split 99.9%, grouped 0.1-degree split 100%, 1 km buffer 99.0%). Treat scores on this
  dataset as inflated; report it as a limitation.
- Planet Amazon dedup: thumbnail-MAD threshold <= 3.0 dropped only 8 of 23,182 tiles (all Normal)
  and kept all rare deforestation tiles (963 train / 207 val / 206 test at the "any deforestation label" level).
- Pillow needs `ImageFile.LOAD_TRUNCATED_IMAGES = True` for the Kaggle wildfire JPGs.
- MobileNetV3-Small Grad-CAM candidate layer: last conv block (`activation_17` in the Keras version;
  in torchvision use `model.features[-1]`).

## Update (second attempt): Planet dedup threshold
- With 24x24 RGB thumbnails, nearest-neighbour mean-abs-difference among 4,000 random Planet tiles ranged
  smoothly from 0.63 (min) to 3.26 (median); no cluster near 0. Visual check: pairs at 0.6-8.0 are all
  different places (homogeneous canopy looks alike). A threshold of 3.0 flagged ~21% of tiles, i.e. wrong.
- Chosen threshold: 0.3 (catches only near-pixel-identical re-saves). The old "<= 3.0" setting depended on
  a different thumbnail definition and is not reused.
