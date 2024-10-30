# LFdiff

Diffusion-based Light Field Synthesis (ECCV Workshop 24)

## Environment
CUDA 11.3

```pip install opencv-python scikit-image h5py matplotlib einops xlwt tensorboard diffusers==0.20.0 imutils==0.5.4 timm==0.6.12 transformers accelerate```




## Data Preparation

Download ([Link](https://pan.baidu.com/s/1LSmRowQE3fG4NW7CCJdiPw), pwd:cbix) processed training data.

You can also manually generate training data by following the instructions below.

For single images, just take step two to obtain the scaled disparity.

### 1.Download datasets

[BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)

[HCInew](https://lightfield-analysis.uni-konstanz.de/)

[UrbanLF-syn](https://github.com/HAWKEYE-Group/UrbanLF)

(Note that the per view png format needs to be converted to .mat to align with the BasicLFSR benchmark for convenient use in step 3.)

[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)




### 2.Use pretrained MDE to obtain monocular disparities

Download pretrained MidaS checkpoint ([Link](https://pan.baidu.com/s/1LSmRowQE3fG4NW7CCJdiPw), pwd:cbix) and put it to the ```/MidaS/weights``` folder.

Put input images (or central views) to the ```/MidaS/input``` folder, the result will be saved to the ```/MidaS/output``` folder.

Run ```MidaS/run.py``` to obtain initial disparity. Then, rescale the resulting disparities to their gt disparity range provided by the original dataset (for single image input, rescale it to customized range such as [-1, 1]).

### 3.Data Preprocess
Create rgb-d data.

```python Generate_Data_for_Training_rgbd.py```

Whether in the training or inference stage, input single images or center view of light fields are cropped into 32x32 spatial resolution patches.


## Training

Set ```state='train'``` and path configs in ```MainCondition.py```

```accelerate launch --multi_gpu MainCondition.py```


## Inference

Download pretrained checkpoint ([Link](https://pan.baidu.com/s/1LSmRowQE3fG4NW7CCJdiPw), pwd:cbix) and put it to the ```CheckpointsCondition``` folder.



