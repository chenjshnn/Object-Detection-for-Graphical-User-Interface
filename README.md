# Object Detection for Graphical User Interface: Old Fashioned or Deep Learning or a Combination?

**Accepted to FSE2020**

*This repository includes all code/pretrained models in our paper, namely Faster RCNN, YOLO v3, CenterNet, Xianyu, REMAUI and our model*

- Paper: Coming soon
- Tool Demo: [Website](http://uied.online), [GitHub](https://github.com/MulongXie/UIED-WebAPP)
- Dataset: Our dataset is based on [Rico](https://interactionmining.org/rico)



## ENVIRONMENT SETUP

All code is tested under Ubuntu 16.04, Cuda 9.0, PyThon 3.5, Pytorch 1.0.1, Nvidia 1080 Ti

### Faster RCNN

```
cd FASTER_RCNN
pip install -r requirements.txt
rm lib/build/*
cd lib & python setup.py build develop
```

### YOLOv3

```
cd PyTorch-YOLOv3
pip install -r requirements.txt
```


### CenterNet

```
cd CenterNet-master
pip install -r requirements.txt
rm models/py_utils/_cpools/build/*
cd models/py_utils/_cpools & python setup.py install --user
```

### Xianyu

Coming soon


### REMAUI

Coming soon


### Our model

Coming soon


## Testing

### FASTER RCNN

```
python demo.py \
--dataset [DATASET] \
--net res101 \
--load_dir results/run \
--pretrained_model_name faster_rcnn.pth \
--cuda \
--vis \
--image_dir [FOLDER-TO-TEST] \
```

DATASET OPTIONS: rico, rico2k, rico10k, ricoDefault, ricoCustomized, ricoText

Put the pretrained model in the folder ("FASTER_RCNN/results/run/res101/{dataset}"). 

- For example, for "ricoCustomized" dataset, place the model in "FASTER_RCNN/results/run/res101/ricoCustomized"

### YOLOv3
```
python detect.py  \
--dataset [DATASET] \
--weights_path result/run/ricotext/yolov3_ckpt.pth \
--image_folder [FOLDER-TO-TEST]
```

DATASET OPTIONS: rico, rico2k, rico10k, rico5box, ricotext

Put the pretrained model in the folder ("PyTorch-YOLOv3/result/run//{dataset}"). 

- For example, for "rico" dataset, place the model in "PyTorch-YOLOv3/result/run/rico"


### CenterNet
```
python demo.py  --cfg_file CenterNet-52-[DATASET] --test_folder [FOLDER-TO-TEST]
```

DATASET OPTIONS: rico, rico2k, rico10k, ricotext

Put the pretrained model in the folder ("CenterNet-master/results/run/CenterNet-52/{dataset}"). 

- For example, for "rico" dataset, place the model in "CenterNet-master/results/run/CenterNet-52/rico"


## ACKNOWNLEDGES

The implementations of Faster RCNN, YOLO v3, CenterNet and REMAUI are based on the following GitHub Repositories. Thank for the works.

- Faster RCNN: https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0

- YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3

- CenterNet: https://github.com/Duankaiwen/CenterNet

- REMAUI: https://github.com/soumikmohianuta/pixtoapp

We implement Xianyu based on their technical blog

- XianYu: https://laptrinhx.com/ui2code-how-to-fine-tune-background-and-foreground-analysis-2293652041/

COCOApi: https://github.com/cocodataset/cocoapi
