# LFW InsightFace Evaluation

Install dependencies:

```powershell
pip install -r lfw_insightface_eval/requirements.txt
```

Run the official LFW protocol (the first run downloads the `buffalo_l` model):

```powershell
python lfw_insightface_eval/evaluate_lfw.py --dataset-root lfw
```

By default, the command evaluates 50 official pairs (25 same-person and 25 different-person pairs, at most 100 image references) to keep CPU load manageable. Use `--max-pairs 6000 --det-size 640` for the full protocol.

The default report is `lfw_insightface_eval/outputs/lfw_metrics.json`. It contains 10-fold thresholds, fold accuracies, mean accuracy, standard deviation, and the number of cached embeddings. Use `--save-pairs` to include every pair score.

Run protocol tests without downloading a model:

```powershell
python -m unittest discover -s lfw_insightface_eval/tests -v
```
