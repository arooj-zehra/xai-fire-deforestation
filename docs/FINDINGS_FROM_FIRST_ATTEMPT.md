# Findings carried over from the first attempt (use in the report)

- Kaggle Wildfire has a spatial shortcut: a k-NN on (lat, lon) alone gets ~99.0-100% accuracy
  (random split 99.9%, grouped 0.1-degree split 100%, 1 km buffer 99.0%). Treat scores on this
  dataset as inflated; report it as a limitation.
- Planet Amazon dedup: thumbnail-MAD threshold <= 3.0 dropped only 8 of 23,182 tiles (all Normal)
  and kept all rare deforestation tiles (963 train / 207 val / 206 test at the "any deforestation label" level).
- Pillow needs `ImageFile.LOAD_TRUNCATED_IMAGES = True` for the Kaggle wildfire JPGs.
- MobileNetV3-Small Grad-CAM candidate layer: last conv block (`activation_17` in the Keras version;
  in torchvision use `model.features[-1]`).
