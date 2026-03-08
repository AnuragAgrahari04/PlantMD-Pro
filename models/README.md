# Models Directory

Place your trained model files here:

- `plantmd_efficientnet.h5` — Trained EfficientNetV2 model
- `labels.json` — Class label mapping

## Training
Run: `python ml_pipeline/train.py --dataset /path/to/plantvillage --output models/`

## Note
Model files are excluded from git (too large). Use DVC for model versioning.
