# DGLNet: Dynamic Global–Local Context Aggregation for Lightweight Facial Expression Recognition

Quick Start Environment setup for DGLNet 
This code is directly related to the manuscript currently submitted to The Visual Computer

## Environment

Python 3.8 + Torch 2.0.0 + CUDA 11.8  
conda create --name DGLNet python=3.8  
conda activate DGLNet

## Hardware:

single RTX 5060 GPU

## Dataset

Since JAFFE and RAF-DB and FER2013 are restricted by their respective licenses, please apply for and download the datasets from their official websites.

**Please place the dataset as follows:**

Dataset/JAFFE/
Dataset/RAF-DB/
Dataset/FER2013/

## Install dependencies

cd DGLNet  
pip install -r requirements.txt

### JAFFE Dataset
https://zenodo.org/record/3451524.
### RAF-DB Dataset

https://www.kaggle.com/datasets/msam-bare/fer2013.

### FER2013 Dataset

http://whdeng.cn/RAF/model1.html/data-set.

## Citations
If you find our work useful in your research, please consider citing:
```python  
@article{
   title = {DGLNet: Dynamic Global–Local Context Aggregation for Lightweight Facial Expression Recognition},
   url={https://github.com/hahappp/DCLNet},
   journal = {{The Visual Computer}}
}
```
