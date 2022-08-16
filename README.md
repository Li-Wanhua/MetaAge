PAP training code
=====================================
## Requirements
- torch 1.2.0 or later versions
- python 3.5.2
- Cuda compilation tools, release 10.1, V10.1.105
- cudnn 6

## Files

### train.py
Codes in this file are used for training the model and validate the results, including data extraction, training loop, etc.

### MetaAge_model.py
Model establishing are major parts of this file.

### loss.py
Loss function is inside this file.

### utilis.py
This file include some less important functions, including model saving, pre-trained model loading, etc.

### resnet.py
The file includes functions of establishing resnet model, which is used for extracting identity features of samples.

## How to train
`CUDA_VISIBLE_DEVICES=4,5 python train.py --pretrained_vgg_path xxx --pretrained_resnet_path xxxx --list_root yyy  --pic_root_dir yyyy`

Explanation of parameters:

- `pretrained_vgg_path` : The path of the pre-trained age estimation model, which is pre-trained on IMDB-WIKI dataset.
- `pretrained_resnet_path` : The path of the pre-trained face recognition model, which is the ResNet-50 version of VGGFace2.
- `list_root` : The path of the dataset image list files, should contain training and validation dataset files.
- `pic_root_dir` : The root path of the dataset images.

### pre-trained models
- Age estimation pre-trained model is needed, which is pre-trained on IMDB-WIKI dataset.
- Face recognition pre-trained model is needed, used for identity features extracting. We employ the ResNet-50 version of VGGFace2.
### training parameters
- **FC_LR = 1e-4** (learning rate of fully-connected parameters)
- **NET_LR = 1e-4** (learning rate of other parameters)
- **BATCH_SIZE = 64** (training batch size)
- **OPTIMIZER = 'adam'** (optimizer choice, adam or sgd)
- **WEIGHT_DECAY = 1e-4** (weight decay, applied only when using SGD)
- **MOMENTUM = 0.9** (momentum, applied only when using SGD)
- **DECAY_RATE = 0.1** (decay rate of learning rate every 10 epoches)
