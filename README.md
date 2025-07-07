## Swin MAE: Masked Autoencoders for Small Datasets

### Introduction
This is a PyTorch implementation of [Swin MAE](https://arxiv.org/abs/2212.13805).

### Usage
1. Install the required environment in "requirements.txt".
2. Open "train.py" and fill in the dataset path. There should be at least one category folder under this path. The data for training is stored in the category folder.
3. Run "train.py".

### Dataset
The paper utilizes two datasets: 
- Parotid gland segmentation dataset;
- [BTCV dataset](https://www.synapse.org/Synapse:syn3193805/wiki/217789).

### License

This code is released under a **custom non-commercial license**.

It is free to use for **non-commercial research and academic purposes only**.  
**Commercial use is strictly prohibited.**  
For commercial licensing inquiries, please contact: xuzian1113@foxmail.com.

### Citation
```
@article{ WOS:001012921200001,
Author = {Xu, Zi'an and Dai, Yin and Liu, Fayu and Chen, Weibing and Liu, Yue and
   Shi, Lifu and Liu, Sheng and Zhou, Yuhang},
Title = {Swin MAE: Masked autoencoders for small datasets},
Journal = {COMPUTERS IN BIOLOGY AND MEDICINE},
Year = {2023},
Volume = {161},
Month = {JUL},
DOI = {10.1016/j.compbiomed.2023.107037},
EarlyAccessDate = {MAY 2023},
Article-Number = {107037},
ISSN = {0010-4825},
EISSN = {1879-0534},
ORCID-Numbers = {Sheng, Liu/0000-0002-5251-2767
   Xu, Zi'an/0000-0002-6374-1805},
Unique-ID = {WOS:001012921200001},
}
```
