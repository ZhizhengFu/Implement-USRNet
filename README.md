# Implement-of-USRNet

## File Structure

| File/Folder | Description |
| --- | --- |
| `options/` | configuration files |
| `kernels/` | pre-trained kernels |
| `model_zoo/` | pre-trained models |
| `models/` | model files |
| `testsets/` | test images |
| `urils` | utility functions |
| `train.py` | training code |
| `test.py` | test on your own images |
| `main_test_bicubic.py` `main_test_realapplication.py` `main_test_table1.py` | test code for images and tables in the paper |
| `USRNet.yaml` | conda environment file |

## Instructions

1. Clone the repository
2. Create a conda environment using the provided `USRNet.yaml` file
3. Prepare the training and testing data:
    - For training, use DIV2K+Flickr2K datasets
    - For testing, use 100 images from ImageNet (you can download from [here](https://drive.google.com/drive/folders/1J4r24ZWWJ6uNqPoIUeowp2Y5zanEnB1M?usp=share_link))
4. Change the paths in `options/train_usrnet.json`:
    - "datasets"/"train"/"dataroot_H": your train data folder
    - "datasets"/"test"/"dataroot_H":  your test data folder 
5. If you have a pre-trained model, change the path in `options/train_usrnet.json`:
    - "path"/"root"/"pretrained_netG": your pre-trained model path
> Optional: you can download pretrained models from [official pretrained models](https://drive.google.com/file/d/1qz8aaYOAMhoKn07VppFjRsDflYpxeVmz/view?usp=sharing) or [my pretrained models]()
> put the pretrained models in `model_zoo/` folder

## Reminder

I use `wandbconfig=True` to log the training process, you can change it to `False` in `test.py` and `train.py` if you don't want to use it.

