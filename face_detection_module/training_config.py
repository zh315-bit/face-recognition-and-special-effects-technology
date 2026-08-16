"""RTMDet-tiny configuration for one-class WIDER FACE training on CPU."""

_base_ = (
    r"C:\Users\huzhuoyang\AppData\Local\Programs\Python\Python311\Lib\site-packages"
    r"\mmdet\.mim\configs\rtmdet\rtmdet_tiny_8xb32-300e_coco.py"
)

data_root = "widerface/"
metainfo = {"classes": ("face",)}

model = dict(bbox_head=dict(num_classes=1))

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/widerface_train.json",
        data_prefix=dict(img="WIDER_train/WIDER_train/images/"),
        metainfo=metainfo,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file="annotations/widerface_val.json",
        data_prefix=dict(img="WIDER_val/WIDER_val/images/"),
        metainfo=metainfo,
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/widerface_val.json",
    metric="bbox",
)
test_evaluator = val_evaluator

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=1)
# Use an ASCII-only absolute path because Windows GBK cannot write the current
# project directory name to MMEngine's last_checkpoint metadata file.
work_dir = r"D:\mmdet_work_dirs\widerface_rtmdet_tiny"
load_from = (
    r"C:\Users\huzhuoyang\.cache\torch\hub\checkpoints"
    r"\rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
)
