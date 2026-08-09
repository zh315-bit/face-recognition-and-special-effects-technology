import unittest

import mmdet_coco_detection


class MmdetCocoDetectionTests(unittest.TestCase):
    def test_parser_uses_coco_model_and_saves_predictions_by_default(self):
        arguments = mmdet_coco_detection.build_parser().parse_args(["--input", "sample.jpg"])

        self.assertEqual(arguments.model, "rtmdet_tiny_8xb32-300e_coco")
        self.assertEqual(arguments.output, "outputs/mmdet_coco")
        self.assertEqual(arguments.score_thr, 0.3)
        self.assertTrue(arguments.save_predictions)
        self.assertIsNone(arguments.max_images)

    def test_parser_accepts_max_images(self):
        arguments = mmdet_coco_detection.build_parser().parse_args(
            ["--input", "celeba", "--max-images", "100"]
        )

        self.assertEqual(arguments.max_images, 100)


if __name__ == "__main__":
    unittest.main()
