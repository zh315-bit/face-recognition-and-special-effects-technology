"""InsightFace .bin verification metrics."""

import numpy as np

def best_threshold(scores, labels, steps=400):
    thresholds = np.linspace(-1.0, 1.0, steps + 1)
    accuracies = [np.mean((scores >= value) == labels) for value in thresholds]
    index = int(np.argmax(accuracies))
    return float(thresholds[index]), float(accuracies[index])

def evaluate_pairs(embeddings, issame, folds=10):
    scores = np.sum(embeddings[0::2] * embeddings[1::2], axis=1)
    labels = np.asarray(issame, dtype=bool)
    indices = np.arange(len(scores)); values = []
    for test in np.array_split(indices, folds):
        train = np.setdiff1d(indices, test); threshold, _ = best_threshold(scores[train], labels[train])
        values.append(float(np.mean((scores[test] >= threshold) == labels[test])))
    return {"accuracy_mean": float(np.mean(values)), "accuracy_std": float(np.std(values))}
