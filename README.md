# Dual Contrast-Driven Deep Multi-View Clustering

This repo contains the code and data associated with our [DCMVC](https://ieeexplore.ieee.org/document/10648641) accepted by **IEEE Transactions on Image Processing 2024**.

## Framework

![Framework Diagram](fig/framework.png)

The overall framework of the proposed DCMVC within an Expectation-Maximization framework. The framework includes: (a) View-specific Autoencoders and Adaptive Feature Fusion Module, which extracts high-level features and fuses them into consensus representations; (b) Dynamic Cluster Diffusion Module, enhancing inter-cluster separation by maximizing the distance between clusters; (c) Reliable Neighbor-guided Positive Alignment Module, improving within-cluster compactness using a pseudo-label and nearest neighbor structure-driven contrastive learning; (d) Clustering-friendly Structure, ensuring well-separated and compact clusters.

## Requirements

hdf5storage==0.1.19

matplotlib==3.5.3

numpy==1.20.1

scikit_learn==0.23.2

scipy==1.7.1

torch==1.8.1+cu111


## Datasets & trained models
The Cora, ALOI-100, Hdigit, and Digit-Product datasets, along with the trained models for these datasets, can be downloaded from [Google Drive](https://drive.google.com/drive/folders/108M1L8fqFk4ZcViZWqQbDe3a2d-uGcXd?usp=drive_link) or [Baidu Cloud](https://pan.baidu.com/s/10vzfz623i4NMx-HslacObQ) password: data.

## Usage

Train a new model：

````python
python train.py
````

Test the trained model:

````python
python test.py
````

## Acknowledgments

Work&Code takes inspiration from [MFLVC](https://github.com/SubmissionsIn/MFLVC), [ProPos](https://github.com/Hzzone/ProPos).

## Citation

If you find our work beneficial to your research, please consider citing:

````latex
@ARTICLE{10648641,
  author={Cui, Jinrong and Li, Yuting and Huang, Han and Wen, Jie},
  journal={IEEE Transactions on Image Processing}, 
  title={Dual Contrast-Driven Deep Multi-View Clustering}, 
  year={2024},
  volume={33},
  number={},
  pages={4753-4764},
  keywords={Feature extraction;Contrastive learning;Reliability;Clustering methods;Task analysis;Data mining;Unsupervised learning;Multi-view clustering;deep clustering;representation learning;contrastive learning},
  doi={10.1109/TIP.2024.3444269}}
````



