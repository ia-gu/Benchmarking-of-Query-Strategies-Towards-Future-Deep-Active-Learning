# Active-Learning-benchmark-tool

The repository contains a Benchmark Tool for Active Learning, SimSiam Pre-training in Python/Pytorch.

Therepository is based on our paper: 

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

Compared Dataset is following: 

- CIFAR10
- ImageNet
- EuroSAT
- OCT
- BrainTumor
- GAPs

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
|  GPU  |  NVIDIA A100  |
|  CPU  |  ??  |
|  CUDA  |  11.6  |
|  Python  |  3.8  |

## Implementation
Clone this repository, modify data_path and implement below

```python main.py```
