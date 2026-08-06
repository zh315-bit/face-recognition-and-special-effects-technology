import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import main


class ImageProcessingTests(unittest.TestCase):
    def test_to_grayscale_converts_bgr_image_to_single_channel(self):
        image = np.array([[[0, 0, 255]]], dtype=np.uint8)

        grayscale = main.to_grayscale(image)

        self.assertEqual(grayscale.shape, (1, 1))
        self.assertEqual(int(grayscale[0, 0]), 76)

    def test_load_image_rejects_unreadable_path(self):
        with TemporaryDirectory() as directory:
            invalid_image = Path(directory) / "not-an-image.txt"
            invalid_image.write_text("not an image", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Could not read image"):
                main.load_image(str(invalid_image))

    def test_main_returns_error_when_image_path_is_missing(self):
        exit_code = main.main([])

        self.assertEqual(exit_code, 2)


if __name__ == "__main__":
    unittest.main()
