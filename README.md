# PV-Defect-Segmentation-DSDINet

Official implementation of DSDINet for Photovoltaic Defect Segmentation (Paper: [DSDINet: Deep Semantic Decoupling and Integration Photovoltaic Defect Segmentation Network])

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PaddleSeg implementation of **DSDINet**, as described in our paper submitted to *Applied Energy*.

> **Paper Title:** DSDINet: Deep Semantic Decoupling and Integration Photovoltaic Defect Segmentation Network

- **[2026-03-16]** Code and configuration files are released for review.

-## 🛠️ Environment Setup

Please ensure you have Python 3.8+ installed. We recommend using Conda. 

1. Install PaddlePaddle and PaddleSeg:
   ```
   pip install paddlepaddle-gpu==2.6.0 -i https://mirror.baidu.com/pypi/simple
   pip install paddleseg==2.8.0
   ```
   
2. Install other dependencies:
   `pip install -r requirements.txt`

-## 📂 Data Preparation
 ```
data/
├── pv_defect/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
 ```
🚀 Training
To train DSDINet, run the following command:
```
python -m paddle.distributed.launch tools/train.py \
       --config configs/dsdinrt/dsdinet_pscde.yml\
       --do_eval \
       --use_vdl \
       --save_interval 50 \
       --save_dir output/test
```

📄 Configuration
The key hyperparameters and architecture settings are defined in:
`configs/dsdinrt/dsdinet_pscde.yml`


📧 Contact
For any questions regarding the code or paper, please contact: 230238499@seu.edu.cn
