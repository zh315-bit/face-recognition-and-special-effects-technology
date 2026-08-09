# 人脸识别与特效技术

本仓库包含课程中的人脸数据可视化、人脸检测和人脸关键点定位实现。

## 人脸检测

`mmdet_coco_detection.py` 使用预训练的 MMDetection RTMDet-tiny COCO
模型进行目标检测，并保存带检测框的可视化图片和预测 JSON 文件。

```powershell
python mmdet_coco_detection.py --input "CelebAMask-HQ\CelebAMask-HQ\CelebA-HQ-img" --max-images 100 --output "outputs\mmdet_coco_celeba100" --device cuda:0
```

> COCO 模型是通用目标检测模型，`person` 类别的结果仅用于完成基线检测与
> 可视化，并不是专用的人脸检测结果。

![MMDetection 检测结果](docs/images/celeba_mmdetection.jpg)

## 人脸关键点定位

`mmpose_face_keypoints.py` 使用预训练的 MMPose 人脸关键点模型，对前一阶段
生成的检测可视化图片进行关键点定位。结果图片和预测 JSON 会保存至
`outputs/mmpose_face_keypoints`。

```powershell
python mmpose_face_keypoints.py --device cuda:0
```

![MMPose 人脸关键点结果](docs/images/celeba_mmpose_keypoints.jpg)

## 测试

```powershell
python -m unittest tests.test_mmdet_coco_detection tests.test_mmpose_face_keypoints
```

## LFW InsightFace 验证

`lfw_insightface_eval/evaluate_lfw.py` 使用 InsightFace `buffalo_l` 预训练模型，依据 LFW 官方 `pairs.txt` 配对协议进行人脸验证。为控制 CPU 负荷，当前基线评测抽取 50 对平衡配对（25 对同人、25 对不同人，最多 100 张图片引用），并使用 10 折交叉验证。

```powershell
python lfw_insightface_eval/evaluate_lfw.py --dataset-root lfw
```

本次快速基线结果为：平均验证准确率 **100.0%**，10 折标准差 **0.0**，各折阈值约为 **0.31**。该结果仅反映 50 对小规模子集的表现，不等同于完整 LFW 6000 对官方评测结果。完整评测可使用：

```powershell
python lfw_insightface_eval/evaluate_lfw.py --dataset-root lfw --max-pairs 6000 --det-size 640
```
