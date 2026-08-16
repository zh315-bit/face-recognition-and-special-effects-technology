from pathlib import Path
import unittest

from export_onnx import onnx_path


class ExportOnnxTests(unittest.TestCase):
    def test_onnx_path_uses_checkpoint_stem(self):
        self.assertEqual(
            onnx_path(Path("runs/epoch_2.pt")),
            Path("runs/epoch_2.onnx"),
        )


if __name__ == "__main__":
    unittest.main()
