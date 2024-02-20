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


# ERROR FASTER_RCNN/lib/model/csrc/cuda/ROIAlign_cuda.cu:5:10: fatal error: THC/THC.h: No such file or directory  #include <THC/THC.h>
```
**Reason**: pytorch >1.10.0 removed this file
**Solution**:
You can replace the cu files in FASTER_RCNN/lib/model/csrc
/cuda/ with files in https://github.com/chenjshnn/Object-Detection-for-Graphical-User-Interface/tree/master/FASTER_RCNN/lib/model/csrc/cuda_support_pytorch_above_1.10

Tested under Ubuntu 22.04, CUDA 12.0, cudnn 8.9.1, pytorch 2.2.0

Alternatively, you can manually update these files following the rules below:

Add these two header files, and remove `#include <THC/THC.h>`
#include <ATen/ceil_div.h>
#include <ATen/cuda/ThrustAllocator.h>

1. For `THCCeilDiv`
- Replace `THCCeilDiv` with `at::ceil_div`
1. For `THCudaCheck`
- Replace `THCudaCheck` with `AT_CUDA_CHECK`
1. For `THCudaFree`
- Replace `THCudaFree(state, mask_dev);` with `c10::cuda::CUDACachingAllocator::raw_delete(mask_dev);`
1. For `THCudaMalloc`
- Replace `THCudaMalloc(state, boxes_num * col_blocks * sizeof(unsigned long long));` with `c10::cuda::CUDACachingAllocator::raw_alloc(boxes_num * col_blocks * sizeof(unsigned long long));`

```
