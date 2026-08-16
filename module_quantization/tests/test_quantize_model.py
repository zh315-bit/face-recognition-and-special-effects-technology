from pathlib import Path
import unittest

from quantize_model import quantized_path


class QuantizeModelTests(unittest.TestCase):
    def test_quantized_path_uses_source_stem(self):
        self.assertEqual(
            quantized_path(Path("runs/last.pt")),
            Path("runs/last_dynamic_int8.pt"),
        )


if __name__ == "__main__":
    unittest.main()
