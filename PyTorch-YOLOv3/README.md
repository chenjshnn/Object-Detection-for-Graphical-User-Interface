# YOLOv3

## Setup

```
cd PyTorch-YOLOv3
pip install -r requirements.txt
```

## Test

```
python detect.py  \
--dataset [DATASET] \
--weights_path result/run/[DATASET]/yolov3_ckpt.pth \
--image_folder [FOLDER-TO-TEST]
```

Dataset options: rico, rico2k, rico10k, rico5box, ricotext

Place the pretrained model in the folder *PyTorch-YOLOv3/result/run/[DATASET]*


**For more information, see https://github.com/eriklindernoren/PyTorch-YOLOv3**
