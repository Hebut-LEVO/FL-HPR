# FLPointcloud_HPR: Personalized Federated Learning Point cloud Human pose recognized

In order to protect the privacy of human body data and solve the problem of recognizing human body posture 
from the non-independent and identically distributed human body point cloud data, this project provides a solution framework.
Please note that this framework is based on personalized federated learning and DCGNN.
## Implemented Algorithms

As initial version, we support the following algoirthms. We are working on more algorithms. 

1. Baseline, train in the client without communication.
2. FedAvg [1].
3. FedProx [2].
4. FedBN [3].
5. FedAP [4].
6. MetaFed [5].

## environment
```
We recommend to use `Python 3.7.1` and `torch 1.7.1` which are in our development environment.
CUDA >= 10.0
Package: glob, h5py, sklearn, plyfile, torch_scatter

## Dataset

Our code supports the following dataset: Hrhp
The sample of dataset are already put into the project and the full dataset will be update later.
If you want to use your own dataset, please modifty `datautil/prepare_data.py` to contain the dataset.

## Usage

1. Modify the file in the scripts
2. `bash run.sh`

## Customization

It is easy to design your own method following the steps:

1. Add your method to `alg/`, and add the reference to it in `alg/algs.py`.

2. Midify `scripts/run.sh` and execuate it.


## Contribution

 Federated learning human posture recognition framework(FL-HPR) is proposed for human posture sensing on nursing robots.

 A federated learning method is introduced for human point cloud segmentation, which can effectively generalize personalized models for local clients.

 A humanjoint estimation method based on dynamic graph edge convolutional network is proposed.

 We implement a demo on dual-arm nursing robots on demonstrate the performance of the human posture recognition in Non-IID data settings and validated the effectiveness of FL-HPR.

## Reference

[1]Y.Wang, Y. Sun, Z. Liu, S. E. Sarma,M.M.Bronstein, J.M. "Solomon,Dynamic graph cnn for learning on point clouds",ACM Transactionson Graphics(tog)38(5)(2019)1–12.

[2] McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.

[3] Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine Learning and Systems 2 (2020): 429-450.

[4] Li, Xiaoxiao, et al. "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization." International Conference on Learning Representations. 2021.

[5] Lu, Wang, et al. "Personalized Federated Learning with Adaptive Batchnorm for Healthcare." IEEE Transactions on Big Data (2022).

[6] Yiqiang, Chen, et al. "MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare." FL-IJCAI Workshop 2022.


## Contact

- Jiaxin Wang: wangjx@hebut.edu.cn

## Contributing


