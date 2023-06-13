# Benchmarking of Query Strategies: Towards Future Deep Active Learning

The repository contains a Benchmark Tool for Active Learning, SimSiam Pre-training in Python/Pytorch.

The repository is based on [our paper (not open now)]()

## Summary

Some of the algorithms currently implemented here include the following:

- Uncertainty Based Sampling
    - [Entropy Sampling](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
    - [Margin Sampling](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)
    - [Least Confidence Sampling](https://ieeexplore.ieee.org/document/6889457)
    - [BALD](https://arxiv.org/abs/1703.02910)
    - [Batch-BALD](https://arxiv.org/abs/1906.08158)
- Diversity Based Sampling
    - Kmeans Sampling
    - [Coreset](https://arxiv.org/abs/1708.00489)
    - [Adversarial BIM](https://arxiv.org/abs/1904.00370)
    - [Adversarial DeepFool](https://arxiv.org/abs/1904.00370)
- Hybrid, other Approaches
    - [BADGE](https://arxiv.org/abs/1906.03671)
    - [FASS](https://openreview.net/forum?id=ByZf6qZuZS)
    - [SIMILAR](https://arxiv.org/abs/2107.00717)
    - [GLISTER](https://arxiv.org/abs/2012.10630)
    - [Cluster Margin Sampling](https://arxiv.org/abs/2107.14263)

<!-- If you want to know the detail of each implementations, refer original paper or [ours](). -->

We use [SimSiam](https://github.com/facebookresearch/simsiam) as the SSL pretraining task. This is based on the idea from [here](https://arxiv.org/abs/2011.10566)

You can get compared dataset from below: 

- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [EuroSAT](https://github.com/phelber/EuroSAT)
- [OCT](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- [BrainTumor](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- [GAPs](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/german-asphalt-pavement-distress-dataset-gaps)
- [KSDD2](https://www.vicos.si/resources/kolektorsdd2/)

In EuroSAT, we split data to train/test = 22,000/5,000. The list we used is in ```materials/eurosat_ids.txt```.

We experiment with 5000 images on OCT, because it is too easy to compare the algorithms.
The list we used is in ```materials/oct_ids.txt```.

BrainTumor dataset is given as MATLAB style, so use ```materials/get_brain_data.py``` to get Image data.

GAPs dataset is given as NPY style, so use ```materials/get_gaps.py``` to get Image data. You need to get a login ID(please refer the official page).

KSDD2 dataset is for segmentation task, so there is no annotation to (OK/NG) on test set.
Thus, We annotate them and the list of NG data ids are in ```materials/ksdd2.csv```.

Finally, this repository is based on [distil](https://github.com/decile-team/distil)



## Environment
Ordinal Implementation setting is below

|  Device |  Detail  |
|  --  |  --  |
|  GPU  |  RTX3090  |
|  CPU  |  Intel (R) Core(TM) i7-10750H  |
|  CUDA  |  11.6  |
|  Python  |  3.8  |

Because of its complex calculation, SimSiam Implementation setting is changed like below

|  Device |  Detail  |
|  --  |  --  |
|  GPU  |  A6000*4  |
|  CPU  |  AMD Ryzen Threadripper PRO 3975WX 32-Cores  |
|  CUDA  |  11.7  |
|  Python  |  3.8  |

ImageNet Implementation setting is below

|  Device |  Detail  |
|  --  |  --  |
|  GPU  |  NVIDIA A100 *4  |
|  CPU  |  AMD EPYC 7742 64-Core Processor  |
|  CUDA  |  11.8  |
|  Python  |  3.8  |

## How to use
Clone this repository, modify data_path and implement below for AL

```python main.py```

Implement below for SSL

```python main_simsiam.py```

## Some more details

We prepare two types of ResNet18. One is the original model given by distil, other is from [here](https://github.com/kuangliu/pytorch-cifar).
This model is for CIFAR dataset because the size of CIFAR images is *32×32*, *7×7* carnel of original model is too big.

In our paper, we did not implement ImageNet experiments, due to time consuming. However, there are few repository which can implement ImageNet×AL experiments, so we prepare it. If you choose ImageNet for dataset and use multi-GPU, the repository automatically implement DDP experiments.