# Object Detection for Graphical User Interface: Old Fashioned or Deep Learning or a Combination?

**Accepted to ESEC/FSE2020**

*This repository includes all code/pretrained models in our paper, namely Faster RCNN, YOLO v3, CenterNet, Xianyu, REMAUI and our model*


## RESOURCE

- Paper: Coming soon
- Tool Demo: [Website](http://uied.online), [GitHub](https://github.com/MulongXie/UIED-WebAPP)
- Dataset: Our dataset is based on [Rico](https://interactionmining.org/rico)

- Pretrained Models: [DropBox](https://www.dropbox.com/sh/xm1ssjkrqep3tah/AADwr4TAaVGak6wx57xuTVZsa?dl=0)


## Code
All code is tested under Ubuntu 16.04, Cuda 9.0, PyThon 3.5, Pytorch 1.0.1, Nvidia 1080 Ti


### Our model
---------

Coming soon



### Faster RCNN
---------

**Setup**

```
cd FASTER_RCNN
pip install -r requirements.txt
rm lib/build/*
cd lib & python setup.py build develop
```

**Test**

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

Dataset options: ricoCustomized, rico2k, rico10k, ricoDefault, ricoText

Put the pretrained model in the folder *FASTER_RCNN/results/run/res101/[DATASET]*


### YOLOv3

**Setup**

```
cd PyTorch-YOLOv3
pip install -r requirements.txt
```

**Test**

```
python detect.py  \
--dataset [DATASET] \
--weights_path result/run/[DATASET]/yolov3_ckpt.pth \
--image_folder [FOLDER-TO-TEST]
```

Dataset options: rico, rico2k, rico10k, rico5box, ricotext

Place the pretrained model in the folder *PyTorch-YOLOv3/result/run/[DATASET]*



### CenterNet

**Setup**


```
cd CenterNet-master
pip install -r requirements.txt
rm models/py_utils/_cpools/build/*
cd models/py_utils/_cpools & python setup.py install --user
```

**Test**

```
python demo.py  \
--cfg_file CenterNet-52-[DATASET] \
--test_folder [FOLDER-TO-TEST]
```

Dataset options: rico, rico2k, rico10k, ricotext

Place the pretrained model in the folder *CenterNet-master/results/run/CenterNet-52/[DATASET]*


### Xianyu

**Setup**

*Tesseract*

```
sudo add-apt-repository -y ppa:alex-p/tesseract-ocr 
sudo apt update
sudo apt install  -y tesseract-ocr
```

*Opencv*

```
python3 -m pip install opencv-python
```

**Test**
```
python3 detect.py --test_folder [FOLDER-TO-TEST]
```



### REMAUI

Coming soon






## ACKNOWNLEDGES

The implementations of Faster RCNN, YOLO v3, CenterNet and REMAUI are based on the following GitHub Repositories. Thank for the works.

- Faster RCNN: https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0

- YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3

- CenterNet: https://github.com/Duankaiwen/CenterNet

- REMAUI: https://github.com/soumikmohianuta/pixtoapp

We implement Xianyu based on their technical blog

- XianYu: https://laptrinhx.com/ui2code-how-to-fine-tune-background-and-foreground-analysis-2293652041/

COCOApi: https://github.com/cocodataset/cocoapi
