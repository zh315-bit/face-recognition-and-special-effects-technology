import tempfile
import unittest
from pathlib import Path

import numpy as np

from lfw_insightface_eval.lfw_protocol import (
    accuracy_at_threshold,
    build_image_path,
    cosine_similarity,
    parse_pairs,
    limit_pairs,
    select_threshold,
    split_folds,
)


class LfwProtocolTests(unittest.TestCase):
    def test_parses_official_same_and_different_records(self):
        with tempfile.TemporaryDirectory() as directory:
            pairs = Path(directory) / "pairs.txt"
            pairs.write_text(
                "2\t1\nAlice\t1\t2\nAlice\t1\tBob\t2\n"
                "Alice\t2\t3\nAlice\t2\tBob\t1\n",
                encoding="utf-8",
            )
            records = parse_pairs(pairs)
        self.assertEqual(len(records), 4)
        self.assertTrue(records[0].same)
        self.assertFalse(records[1].same)
        self.assertEqual(records[1].left, ("Alice", 1))
        self.assertEqual(records[1].right, ("Bob", 2))

    def test_builds_lfw_image_path(self):
        self.assertEqual(
            build_image_path(Path("lfw"), ("Ada_Lovelace", 7)),
            Path("lfw") / "lfw" / "Ada_Lovelace" / "Ada_Lovelace_0007.jpg",
        )

    def test_similarity_threshold_and_accuracy(self):
        self.assertAlmostEqual(cosine_similarity(np.array([1.0, 0.0]), np.array([2.0, 0.0])), 1.0)
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([True, True, False, False])
        threshold = select_threshold(scores, labels)
        self.assertGreaterEqual(threshold, 0.2)
        self.assertEqual(accuracy_at_threshold(scores, labels, threshold), 1.0)

    def test_splits_pairs_into_disjoint_folds(self):
        scores = np.arange(8, dtype=float)
        labels = np.array([True, False] * 4)
        folds = split_folds(scores, labels, 4)
        self.assertEqual(len(folds), 4)
        self.assertEqual(sum(len(test_scores) for _, test_scores, _, _ in folds), 8)

    def test_limits_pairs_with_balanced_labels(self):
        records = parse_pairs(Path("lfw") / "pairs.txt")
        limited = limit_pairs(records, 100)
        self.assertEqual(len(limited), 100)
        self.assertEqual(sum(record.same for record in limited), 50)
        self.assertNotEqual(limited[0].same, limited[1].same)


if __name__ == "__main__":
    unittest.main()
