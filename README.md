# MaskAttn-UNet: A Mask Attention-Driven Framework for Universal Low-Resolution Image Segmentation
This repository provides an innovative segmentaion framework called **MaskAttn-UNet** designed specifically to address various challenges posed by **128 x 128 low-resolution images** in applications like robotics, augmented reality, and large-scale scene understanding.

By integrating a mask attention module into the traditional U-Net architecture, our MaskAttn-Unet not only maintains a moderate memory footprint during inference but also taking care of the fine-grained local details, particularly suitable for resource-constained environments, building upon our paper:
>[MaskAttn-UNet: A Mask Attention-Driven Framework for Universal Low-Resolution Image Segmentation]
>


## Overview
Paper is implemented with official pytorch
![Overview Image](figures/overview.png?raw=true "Overview of the proposed MaskAttn-UNet")

This high-level overview figure clearly illustrates the detailed architecture of the proposed MaskAttn-UNet. Specifically, **section (a)** outlines the comprehensive encoder-decoder structure based on the traditional U-Net model, **section (b)** presents the internal structure of the Mask Attention Module, and **section (c)** provides a detailed illustration of our Mask-Attn UNet with the multi-scale encoder-decoder design.

## Github Repository Structrure
```
MaskUnet/
│
├── code/
│   ├── ade20k/
│   │   ├── ade_instance.py    # Python file for Instance Segmentation on ADE20K dataset
│   │   ├── ade_panoptic.py    
│   │   └── ade_semantic.py
│   │
│   ├── cityscapes/
│   │   ├── city_instance.py
│   │   ├── city_panoptic.py   # Python file for Panoptic Segmentation on Cityscapes dataset 
│   │   └── city_semantic.py
│   │
│   └── coco/
│       ├── coco_instance.py  
│       ├── coco_panoptic.py
│       └── coco_semantic.py   # Python file for Semantic Segmentation on COCO dataset
│
├── data/                      # Dataset downloader files
│   ├── ADEK/
│   │   └── data_download.py 
│   │
│   ├── Cityscapes/
│   │   └── data_download.py
│   │
│   └── COCO/
│       └── data_download.py
│
├── figures/                   # Figures for README and others
│   ├── overview.png
│   └── Segmentation_performance.png
│
├── .gitignore                 
├── .gitignore.swp           
├── README.md               
└── requirements.txt           # Python dependencies and environment requirements


```
## Requirements
* **Python** 3.8+
* **PyTorch** 2.0.0+
* **torchvision** 0.15.0+
*  **NumPy**,**scikit-image**,**matplotlib**,etc
  
Please go to the `requirement.txt` file to check all dependencies.

Or run the following code to install:
```
pip install -r requirements.txt
```
## Data Downloading
First, make sure you have successfully cloned the repo, then go direct to the `data` folder by
```
cd data
```

### ADE20K Dataset
Open the notebook file `ade_download.ipynb` and follow the instructions provided. You'll find two options for downloading the dataset:

* Using Command Line: Execute commands directly in your terminal.

* Using a Python Script: Log into your account and run the provided script
  
### COCO Dataset
Open and execute the notebook file `coco_download.ipynb`. This notebook will automatically download the required dataset files and create a subset for subsequent use.

### Cityscapes Dataset
Open `cityscapes_download.ipynb` to download the two packages: `gtFine_trainvaltest.zip (241MB)` and `leftImg8bit_trainvaltest.zip (11GB)`.

*Note: Recently, we found that the CITYSCAPES official website may be blocking scripted logins and downloads. If you encounter issues with both methods, please download the dataset manually from the [official Cityscapes website](https://www.cityscapes-dataset.com/downloads/).*


## Training
*Note: The hyperparameters within the code folders under the three dataset (eg: ade_instance.py, ade_panoptic.py, and ade_semantic.py) may not reflect the latest configuration. Please refer to the Final Notes and consult with PB for the most current hyperparameter settings.*

To train the MaskAttn-UNet model and obtain evaluation metrics for Semantic, Instance, or Panoptic segmentation, execute the relevant Python script as like follows:
```
python ade_semantic.py
```

## Results
![alt text](figures/comparison.png?raw=true "Comparison with SOTA Models")

MaskAttn-UNet is outperforming most of the state-of-the-art models for panoptic segmentation. The best results are highlighted in **bold**, and the second best are **underlined**.

![alt text](figures/Segmentation_performance.png?raw=true "Error rate of different methods")

Segmentation performance of MaskAttn-UNet on different fractions of the *panoptic_train2017* dataset. These trends illustrates the significant data efficiency of MaskAttn-UNet, making it a practical choice for circumstances with limited annotated data. 

