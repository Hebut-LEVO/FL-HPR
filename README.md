# PersonalizedFL: Personalized Federated Learning Codebase

An easy-to-learn, easy-to-extend, and for-fair-comparison codebase based on PyTorch for federated learning (FL). 
Please note that this repository is designed mainly for research, and we discard lots of unnecessary extensions for a quick start.
Example usage: when you want to recognize activities of different persons without accessing their privacy data; when you want to build a model based on multiple patients' data but not access their own data.

## Implemented Algorithms

**Note:** The code of [FedCLIP](https://arxiv.org/abs/2302.13485v1) will be released soon. Please stay tuned.

As initial version, we support the following algoirthms. We are working on more algorithms. 

1. Baseline, train in the client without communication.
2. FedAvg [1].
3. FedProx [2].
4. FedBN [3].
5. FedAP [4].
6. MetaFed [5].

## Installation

```
git clone https://github.com/microsoft/PersonalizedFL.git
cd PersonalizedFL
```
We recommend to use `Python 3.7.1` and `torch 1.7.1` which are in our development environment. 
For more environmental details and a full re-production of our results, please refer to `luwang0517/torch10:latest` (docker) or `jindongwang/docker` (docker).

## Dataset

Our code supports the following dataset:Hrhp
* [OrganC-MNIST](https://wjdcloud.blob.core.windows.net/dataset/cycfed/medmnistC.tar.gz)

If you want to use your own dataset, please modifty `datautil/prepare_data.py` to contain the dataset.

## Usage

1. Modify the file in the scripts
2. `bash run.sh`

## Customization

It is easy to design your own method following the steps:

1. Add your method to `alg/`, and add the reference to it in `alg/algs.py`.

2. Midify `scripts/run.sh` and execuate it.


## Contribution

The toolkit is under active development and contributions are welcome! Feel free to submit issues and PRs to ask questions or contribute your code. If you would like to implement new features, please submit a issue to discuss with us first.

## Reference

[1] McMahan, Brendan, et al. "Communication-efficient learning of deep networks from decentralized data." Artificial intelligence and statistics. PMLR, 2017.

[2] Li, Tian, et al. "Federated optimization in heterogeneous networks." Proceedings of Machine Learning and Systems 2 (2020): 429-450.

[3] Li, Xiaoxiao, et al. "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization." International Conference on Learning Representations. 2021.

[4] Lu, Wang, et al. "Personalized Federated Learning with Adaptive Batchnorm for Healthcare." IEEE Transactions on Big Data (2022).

[5] Yiqiang, Chen, et al. "MetaFed: Federated Learning among Federations with Cyclic Knowledge Distillation for Personalized Healthcare." FL-IJCAI Workshop 2022.


## Contact

- Jiaxin Wang: wangjx@hebut.edu.cn

## Contributing


