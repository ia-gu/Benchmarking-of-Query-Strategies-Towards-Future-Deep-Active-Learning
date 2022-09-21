# Active-Learning-benchmark-tool

The repository contains a Benchmark Tool for Active Learning, SimSiam Pre-training in Python/Pytorch.

The repository is based on our paper: 

## Summary

Some of the algorithms currently implemented here include the following:

- Uncertainty Based Sampling
    - [Entropy Sampling](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)
    - [Margin Sampling](http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf)
    - [Least Confidence Sampling](https://ieeexplore.ieee.org/document/6889457)
    - [BALD](https://arxiv.org/pdf/1703.02910.pdf)
    - [Batch-BALD](https://arxiv.org/pdf/1906.08158.pdf)
- Diversity Based Sampling
    - [Kmeans Sampling]
    - [Coreset](https://openreview.net/pdf?id=H1aIuk-RW)
    - [Adversarial BIM](https://arxiv.org/pdf/1904.00370.pdf)
    - [Adversarial DeepFool](https://arxiv.org/pdf/1904.00370.pdf)
- Hybrid, other Approaches
    - [BADGE](https://arxiv.org/pdf/1906.03671.pdf)
    - [FASS](http://proceedings.mlr.press/v37/wei15.pdf)
    - [SIMILAR](https://arxiv.org/pdf/2107.00717.pdf)
    - [GLISTER](https://arxiv.org/pdf/2012.10630.pdf)
    - [Cluster Margin Sampling](https://arxiv.org/pdf/2107.14263.pdf)

If you want to know the detail of each implementations, refer original paper or [ours]().

SSL implemented here is [SimSiam](https://github.com/facebookresearch/simsiam)

You can get compared dataset from below: 

- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](https://image-net.org/challenges/LSVRC/2012/index)
- [EuroSAT](https://github.com/phelber/EuroSAT)
- [OCT](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)
- [BrainTumor](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- [GAPs](https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/german-asphalt-pavement-distress-dataset-gaps)

We experiment with 5000 images on OCT, the list we used is in oct_ids.txt

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
If you do not use SSL, you can modify these.

We prepare two types of ResNet18. One is the original model given by distil, other is from [here](https://github.com/kuangliu/pytorch-cifar).
This model is for CIFAR dataset because CIFAR is 32*32 image, 7*7 carnel of original model is too big.

