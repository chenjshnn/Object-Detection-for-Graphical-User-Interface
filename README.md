# Object Detection for Graphical User Interface: Old Fashioned or Deep Learning or a Combination?

**Accepted to ESEC/FSE2020**

*This repository includes all code/pretrained models in our paper, namely Faster RCNN, YOLO v3, CenterNet, Xianyu, REMAUI and our model*


## RESOURCE

- Paper: [arXiv](https://arxiv.org/abs/2008.05132), [ACM](https://dl.acm.org/doi/abs/10.1145/3368089.3409691)
- Video: [YouTube](https://www.youtube.com/watch?v=KFFp81N6zlg&t=166s&ab_channel=ACMSIGSOFT)
- Tool Demo: [Website](http://uied.online), [GitHub](https://github.com/MulongXie/UIED-WebAPP)
- Dataset: Our dataset is based on [Rico](https://interactionmining.org/rico)

- Pretrained Models: [DropBox](https://www.dropbox.com/sh/xm1ssjkrqep3tah/AADwr4TAaVGak6wx57xuTVZsa?dl=0)
- Tool: http://uied.online


## CODE
All code is tested under Ubuntu 16.04, Cuda 9.0, PyThon 3.5, Pytorch 1.0.1, Nvidia 1080 Ti


### Our model

See [GitHub](https://github.com/MulongXie/UIED)


### Baselines

See the corresponding folder in this repository. Each folder contains an individual README file.

[Faster RCNN](./FASTER_RCNN), [YOLOv3](./PyTorch-YOLOv3), [CenterNet](./CenterNet-master), [Xianyu](./Xianyu)

For REAMUI, see [pix2app](https://github.com/soumikmohianuta/pixtoapp)


## ACKNOWNLEDGES

The implementations of Faster RCNN, YOLO v3, CenterNet and REMAUI are based on the following GitHub Repositories. Thank for the works.

- Faster RCNN: https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0

- YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3

- CenterNet: https://github.com/Duankaiwen/CenterNet

- REMAUI: https://github.com/soumikmohianuta/pixtoapp

We implement Xianyu based on their technical blog

- XianYu: https://laptrinhx.com/ui2code-how-to-fine-tune-background-and-foreground-analysis-2293652041/

COCOApi: https://github.com/cocodataset/cocoapi


# Citation

```
@inproceedings{chen2020object,
  title={Object Detection for Graphical User Interface: Old Fashioned or Deep Learning or a Combination?},
  author={Chen, Jieshan and Xie, Mulong and Xing, Zhenchang and Chen, Chunyang and Xu, Xiwei, Zhu, Liming and Guoqiang Li},
  booktitle={Proceedings of the 2020 28th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering},
  year={2020},
  publisher = "ACM",
  address = "New York, NY",
  doi = "10.1145/3368089.3409691",
}
```
