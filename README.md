# Face Recognition and Special Effects Technology

This repository contains the course implementation for face data visualization,
face detection, and face landmark localization.

## Face Detection

`mmdet_coco_detection.py` uses the pretrained MMDetection RTMDet-tiny COCO
model to visualize detections and save prediction JSON files.

```powershell
python mmdet_coco_detection.py --input "CelebAMask-HQ\CelebAMask-HQ\CelebA-HQ-img" --max-images 100 --output "outputs\mmdet_coco_celeba100" --device cuda:0
```

> The COCO model is a general object detector. Its `person` results are a
> baseline visualization and are not a face-specific detector.

![MMDetection result](docs/images/celeba_mmdetection.jpg)

## Face Landmark Localization

`mmpose_face_keypoints.py` uses the pretrained MMPose face landmark model on
the preceding detection visualization images. It saves landmark images and
prediction JSON files under `outputs/mmpose_face_keypoints`.

```powershell
python mmpose_face_keypoints.py --device cuda:0
```

![MMPose face landmark result](docs/images/celeba_mmpose_keypoints.jpg)

## Tests

```powershell
python -m unittest tests.test_mmdet_coco_detection tests.test_mmpose_face_keypoints
```
