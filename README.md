# Active-Learning-benchmark-tool

The repository contains a Benchmark Tool for Active Learning, SimSiam Pre-training in Python/Pytorch.

The repository is based on [our paper]()

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

If you want to know the detail of each implementations, refer original paper or [ours]().

SSL implemented here is [SimSiam](https://github.com/facebookresearch/simsiam). This is based on the idea from [here](https://arxiv.org/abs/2011.10566)

You can get compared dataset from below: 

- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](https://image-net.org/challenges/LSVRC/2012/index)
- [EuroSAT](https://github.com/phelber/EuroSAT)
- [OCT](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- [BrainTumor](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- [GAPs](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/german-asphalt-pavement-distress-dataset-gaps)

We experiment with 5000 images on OCT, because it is too easy to compare the algorithms.
The list we used is in ```oct_ids.txt```

The repository is forked from [distil](https://github.com/decile-team/distil)



## Environment
Ordinal Implementation setting is below

|  Device |  Detail  |
|  --  |  --  |
|  GPU  |  RTX3090  |
|  CPU  |  Intel (R) Core(TM) i7-10750H  |
|  CUDA  |  11.6  |
|  Python  |  3.8  |

ImageNet Implementation setting is below

|  Device |  Detail  |
|  --  |  --  |
|  GPU  |  NVIDIA A100 *4  |
|  CPU  |  AMD EPYC 7742 64-Core Processor  |
|  CUDA  |  11.6  |
|  Python  |  3.8  |

## How to use
Clone this repository, modify data_path and implement below for AL

```python main.py```

Implement below for SSL

```python main_simsiam.py```

## Some more details
```get_brain_data.py``` and ```get_gaps.py``` is made because we need image data when we implement SimSiam.

We prepare two types of ResNet18. One is the original model given by distil, other is from [here](https://github.com/kuangliu/pytorch-cifar).
This model is for CIFAR dataset because the size of CIFAR images is *32×32*, *7×7* carnel of original model is too big.

In our paper, we did not implement ImageNet experiments, due to time consuming. However, there are few repository which can implement ImageNet×AL experiments, so we prepare it. If you choose ImageNet for dataset and use multi-GPU, the repository automatically implement DDP experiments.