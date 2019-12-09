# Environment
* Python: 3.6
* OS: Ubuntu 16.04 


# Package
* cudnn: 7.6.4
* cuda (&toolkit): 9.2_0


# Python-Package
* cv2(opencv-python)==3.4.2
* scikit-image==0.15.0
* scikit-learn==0.21.3
* h5py==2.8.0
* imageio==2.6.1
* Keras==2.2.4
* Keras-Applications==1.0.8
* Keras-Preprocessing==1.1.0
* matplotlib==3.1.1
* numpy==1.16.4
* Pillow==6.2.1
* protobuf==3.10.1
* scikit-image==0.15.0
* scikit-learn==0.21.3
* scipy==1.3.1
* tensorboard==1.12.2
* tensorflow==1.12.0


# Commands set
## How to train & validate
```
python3 train.py                         # GPU-version
CUDA_VISIBLE_DEVICES="" python3 train.py # CPU-version
```

## How to test
```
python3 test.py                         # GPU-version
CUDA_VISIBLE_DEVICES="" python3 test.py # CPU-version
```

## How to generate heatmap image
```
python3 grad_cam.py                         # GPU-version
CUDA_VISIBLE_DEVICES="" python3 grad_cam.py # CPU-version
```

## How to use tensorboard
```
LC_ALL=C tensorboard --logdir='./tensorboard'

# then enter ip:1006 to see the what's going on
```


# Dir/Files usage
| File/Dir Name  |  Usage  |
|---|---|
| dac_mode.py  | model  |
| train.py  | train+val  |
| test.py  | test  |
| grad_cam.py  | generate grad_cam image  |
| cam_imgs/  | grad_cam images  |


# Example
![](https://github.com/r06725028/2019-tsmc-dac/blob/master/example_images/new_img_1.jpg)
