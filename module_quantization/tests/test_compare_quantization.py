import unittest

from compare_quantization import percent_change


class CompareQuantizationTests(unittest.TestCase):
    def test_percent_change_reports_size_reduction(self):
        self.assertEqual(percent_change(100, 20), -80.0)


if __name__ == "__main__":
    unittest.main()
