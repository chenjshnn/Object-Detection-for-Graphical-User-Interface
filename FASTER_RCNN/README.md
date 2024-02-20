source github: https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0

# Faster RCNN

## Setup

```
cd FASTER_RCNN
pip install -r requirements.txt
rm -r lib/build/*
cd lib & python setup.py build develop
```

## Test

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

## Common issues and solutions
See [DL_setup_troubleshootiing.md](../DL_setup_troubleshootiing.md)

**For more information, see https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0**


