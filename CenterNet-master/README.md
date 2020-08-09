# CenterNet

## Setup


```
pip install -r requirements.txt
rm models/py_utils/_cpools/build/*
cd models/py_utils/_cpools & python setup.py install --user
```

## Test

```
python demo.py  \
--cfg_file CenterNet-52-[DATASET] \
--test_folder [FOLDER-TO-TEST]
```

Dataset options: rico, rico2k, rico10k, ricotext

Place the pretrained model in the folder *CenterNet-master/results/run/CenterNet-52/[DATASET]*

**For more information, see https://github.com/Duankaiwen/CenterNet**
