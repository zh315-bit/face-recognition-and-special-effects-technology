# Face Recognition Training Module

This folder contains the ResNet50-IR ArcFace training implementation, data
configuration, LFW validation code, and validation artifacts from the
completed epoch-2 model.

## Main Files

- `train.py`: ResNet50-IR and ArcFace training entry point.
- `model.py`: ResNet50-IR encoder definition.
- `arcface.py`: ArcFace margin head used by the training loss.
- `config.py`: RecordIO data discovery and class-count configuration helpers.
- `recordio.py`: InsightFace RecordIO dataset loader.
- `evaluate_lfw.py`: LFW verification evaluation entry point.
- `reports/lfw_100_metrics.json`: 100-pair LFW result: accuracy `0.95`, ROC-AUC `0.9796`.
- `reports/lfw_100_roc.png`: ROC curve for the same evaluation.
- `training_loss_status.md`: explains why no historical loss file is available.

## Run

From the project root, install dependencies and run training:

```powershell
py -m pip install -r face_recognition/requirements.txt
py -m face_recognition.train --data-root ./faces_emore --output-dir runs/resnet50_arcface --device cuda
```

To reproduce the LFW validation, pass the trained checkpoint and LFW root to
the evaluator. The training checkpoint itself remains in `faces_emore/runs/`
and is not duplicated in this source-and-results module.
