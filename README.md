# Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts

This repository presents the experiments of the paper:

[Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts](http://papers.nips.cc/paper/9445-uncertainty-on-asynchronous-time-event-prediction.pdf)<br>
Bertrand Charpentier, Daniel Zügner, Stephan Günnemann<br>
Conference on Neural Information Processing Systems (NeurIPS), 2020.

[[Paper](https://arxiv.org/pdf/2006.09239.pdf)]

![Diagram](diagram-1.png?raw=true "Diagram")

## Requirements

To install requirements:

```setup
conda env create -f environment.yaml
conda activate posterior-network
conda env list
```

## Training & Evaluation

To train the model(s) in the paper, run one jupyter notebook in the folder `notebooks`. All parameter are described.

To dowload the datasets, follow the following links:
- [2DGaussians](https://ln2.sync.com/dl/80d65ce00#tk74rers-cmuhhp64-vp2zp28m-mm7zsmsn) vs [anomalous2D](https://ln2.sync.com/dl/1940f4fd0#vt62pk6j-gmpt5yw6-c4w922rb-7jkw4g22)
- [Segment (No sky)](https://ln2.sync.com/dl/808d7d3d0#vsyw43e6-qi3t65qk-x86u838c-nehzxrc3) vs [Segment (Sky only)](https://ln2.sync.com/dl/52ea91a20#rppt45jy-wtmhpp52-k6haa7w2-5p3k5zfg)
- [SensorlessDrive (No 10, 11)](https://ln2.sync.com/dl/c41a8a050#i4gbn3wt-a6qjwbgd-m4ch8g6a-eacp6bh6) vs [SensorlessDrive (10, 11 only)](https://ln2.sync.com/dl/8b09e0d40#jmagkizd-dpguftcv-tfx4jktk-meqi5hju)
- [MNIST](https://ln2.sync.com/dl/315769850#uhd888js-tqv8xn4u-264x5xhr-ey6iqfaw) vs [FashionMNIST](https://ln2.sync.com/dl/0220ee6f0#2hd374ka-bt8jf94g-uy7hy5jt-2uxideru) / [KMNIST](https://ln2.sync.com/dl/703efb2a0#gfyih8rc-3t6s37cj-8mhmg3qx-zqj6hzqu)
- [CIFAR10](https://ln2.sync.com/dl/c18096180#ai7q5kjw-2a4ebpy2-gbemwuan-bbp45sxh) vs [SVHN](https://ln2.sync.com/dl/4e3742920#rmt7apcw-3j2yursi-gpuktnmw-2e9tsv8y)

## Pre-trained Models

You can find pre-trained models in the folder `saved_models`.

## Cite
Please cite our paper if you use the model or this code in your own work:

```
@incollection{postnet,
title = {Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts},
author = {Charpentier, Bertrand, Daniel Z\"{u}gner and G\"{u}nnemann, Stephan},
booktitle = {Advances in Neural Information Processing Systems 33},
year = {2020},
publisher = {Curran Associates, Inc.},
}
```