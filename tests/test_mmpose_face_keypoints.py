import unittest

import mmpose_face_keypoints


class MMPoseFaceKeypointsTests(unittest.TestCase):
    def test_parser_uses_previous_celeba_outputs_by_default(self):
        arguments = mmpose_face_keypoints.build_parser().parse_args([])

        self.assertEqual(arguments.input, "outputs/mmdet_coco_celeba100/vis")
        self.assertEqual(arguments.output, "outputs/mmpose_face_keypoints")
        self.assertEqual(arguments.device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
