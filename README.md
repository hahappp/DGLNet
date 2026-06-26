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
http://whdeng.cn/RAF/model1.html/data-set.

### FER2013 Dataset
https://www.kaggle.com/datasets/msam-bare/fer2013.

## Occlusion-RAF-DB and Pose-RAF-DB Protocols

This project does not redistribute the RAF-DB images or any third-party dataset files. To reproduce the occlusion and pose robustness experiments, users should first obtain the original RAF-DB dataset from its official source and comply with its license and usage terms.

The original authors provide the RAF-DB occlusion/pose subset lists and labels at:
👉 [Challenge-condition-FER-dataset/RAF_DB_dir](https://github.com/kaiwang960112/Challenge-condition-FER-dataset/tree/master/RAF_DB_dir)

Please download these list/label files from the original repository. Our code only reads externally obtained files and does not claim ownership of RAF-DB or the derived subset annotations.

If you evaluate models using these protocols, please ensure you cite the corresponding paper:

```bibtex
@article{wang2020region,
  title={Region attention networks for pose and occlusion robust facial expression recognition},
  author={Wang, Kai and Peng, Xiaojiang and Yang, Jianfei and Meng, Debin and Qiao, Yu},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={4057--4069},
  year={2020},
  publisher={IEEE}
}
```

## Quick Start

### 1. Install dependencies

```bash
cd DGLNet
pip install -r requirements.txt
```
### 2. Prepare datasets
Please download the datasets from their official sources and organize them as follows:
Dataset/
├── RAF-DB/
│   ├── train/
│   └── val/
├── FER2013/
│   ├── train/
│   └── val/
└── JAFFE/
    ├── train/
    └── val/
Due to dataset license restrictions, we do not redistribute the original images.

### 3. Train DGLNet on RAF-DB
```bash
python Train/train_RAF.py --train-path Dataset/RAF-DB/train --val-path Dataset/RAF-DB/val --batch-size 64 --epochs 120 --lr 5e-5
```

# To ensure full reproducibility, we explicitly list the core hyper-parameters and configurations used for training DGLNet on the dataset, which align exactly with the settings reported in our manuscript:
* **Optimizer:** AdamW (`lr=5e-5`, `weight_decay=0.05`)
* **Learning Rate Scheduler:** Cosine Annealing (`min_lr=1e-6`, `warmup_epochs=10`)
* **Loss Function:** CrossEntropy (with `label_smoothing=0.1`)
* **Training Epochs:** 120 epochs
* **Batch Size:** 64
* **Gradient Clipping:** `max_norm=1.0`
* **Drop Path Rate:** 0.2
* **Data Augmentation:** RandAugment (`num_ops=2`, `magnitude=12`), Random Erasing (`p=0.5`), ColorJitter, and RandomRotation.
* **Random Seed:** 42, 2026, 3407 (To ensure robustness and rule out coincidence, experiments were rigorously verified across multiple random seeds. The default seed in the provided scripts is 42.)

## Edge Deployment & ONNX Inference Demo

To explicitly demonstrate the deployment potential of DGLNet on resource-constrained edge devices, we provide scripts to export the PyTorch model to the **ONNX** format and run a lightweight static image inference demo. This inference script is entirely independent of the PyTorch framework and relies solely on the lightweight `onnxruntime` engine.

**1. Download ONNX Weights:**
To keep the repository lightweight, the exported `.onnx` industrial deployment weights are hosted in the GitHub Releases. 
* Please download `dglnet_rafdb.onnx` from the [Releases](#) page and place it in the root directory (or update the path in the script).

**2. Export PyTorch Model to ONNX (Optional) and Run Lightweight Inference Demo:**
This script performs image preprocessing, feature extraction, and classification using only numpy and onnxruntime.If you wish to export the model yourself, you can run the following script:
```bash
python Tools/export_onnx.py
python Tools/onnx_image_infer.py
```

## Citations
If you find our work useful in your research, please consider citing:
```python  
@article{
   title = {DGLNet: Dynamic Global–Local Context Aggregation for Lightweight Facial Expression Recognition},
   url={https://github.com/hahappp/DGLNet},
   journal = {{The Visual Computer}}
}
```
