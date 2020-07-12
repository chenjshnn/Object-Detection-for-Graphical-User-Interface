Source github: 
https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0
https://github.com/Duankaiwen/CenterNet

Environment: cuda 9.0, python 3.5, pytorch 1.0.1, ubuntu 16.04, Nvidia 1080 Ti



# FASTER RCNN
```
pip install -r requirements.txt
rm lib/build/*
cd lib & python setup.py build develop
```

train.sh -- use to train the model
test.sh -- use to test the test dataset
demo.sh -- use to detect objects in imgs


# CENTERNET
```

pip install -r requirements.txt
rm models/py_utils/_cpools/build/*
cd models/py_utils/_cpools & python setup.py install --user

```

demo.sh -- use to detect objects in imgs


*Please setup FASTER_RCNN first*


# ERROR: _cannot import name '_mask'
```
unzip cocoapi-master.zip
cd cocoapi-master/PythonAPI
sudo make

```
replace the old *pycocotools* FOLDER in CenterNet-master & FASTER_RCNN with the new one