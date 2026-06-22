# DGLNet: Dynamic Global–Local Context Aggregation for Lightweight Facial Expression Recognition

Quick Start Environment setup for DGLNet 
This code is directly related to the manuscript currently submitted to The Visual Computer

## Environment

Python 3.8 + Torch 2.0.0 + CUDA 11.8  
conda create --name DGLNet python=3.8  
conda activate DGLNet

## Hardware:

NVIDIA RTX 5060 Ti GPU (16GB)

## Dataset

Since the facial expression recognition datasets are restricted by their respective official licenses, we cannot distribute the images directly. Please apply for access and download them from their official repositories.

## Please place the dataset as follows:

Dataset/JAFFE/
Dataset/RAF-DB/
Dataset/FER2013/

### JAFFE Dataset
https://zenodo.org/record/3451524.
### RAF-DB Dataset

https://www.kaggle.com/datasets/msam-bare/fer2013.

### FER2013 Dataset

http://whdeng.cn/RAF/model1.html/data-set.

## Install dependencies

cd DGLNet  
pip install -r requirements.txt

# To ensure full reproducibility, we explicitly list the core hyper-parameters and configurations used for training DGLNet on the dataset, which align exactly with the settings reported in our manuscript:
Optimizer: AdamW (`lr=5e-5`, `weight_decay=0.05`)
Learning Rate Scheduler: Cosine Annealing (`min_lr=1e-6`, `warmup_epochs=10`)
Loss Function: CrossEntropy (with `label_smoothing=0.1`) 
Training Epochs: 120 epochs 
Batch Size: 64
Gradient Clipping: `max_norm=1.0`
Drop Path Rate: 0.2
Data Augmentation: RandAugment (`num_ops=2`, `magnitude=12`), Random Erasing (`p=0.5`), ColorJitter, and RandomRotation.
Random Seed: 42, 2026, 3407 (To ensure robustness and rule out coincidence, experiments were rigorously verified across multiple random seeds. The default seed in the provided scripts is 42.)

## Citations
If you find our work useful in your research, please consider citing:
```python  
@article{
   title = {DGLNet: Dynamic Global–Local Context Aggregation for Lightweight Facial Expression Recognition},
   url={https://github.com/hahappp/DGLNet},
   journal = {{The Visual Computer}}
}
```
