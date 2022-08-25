# Improving Knowledge-aware Recommendation with Multi-level Interactive Contrastive Learning
This is our Pytorch implementation for the paper:
> Ding Zou, Wei Wei, Ziyang Wang, Xian-Ling Mao, Feida Zhu, Rui Fang, and Dangyang Chen (2022). Improving Knowledge-aware Recommendation with Multi-level Interactive Contrastive Learning
[Paper in Arxiv](https://arxiv.org/pdf/2208.10061.pdf), In CIKM 2022

## Introduction
Knowledge-aware Recommender System with Multi-level Interactive Contrastive Learning (KGIC) is a knowledge-aware recommendation solution based on GNN and Contrastive Learning. 
KGIC combines multi-order CF with KG to construct local and non-local graphs for fully exploring external knowledge, and proposes a multi-level interactive contrastive mechanism 
tailored for knowledge-aware recommendation (intra- and inter-graph levels) for a sufficient and coherent information
utilization in CF and KG.

## Requirement
The code has been tested running under Python 3.7.9. The required packages are as follows:
- pytorch == 1.5.0
- numpy == 1.15.4
- sklearn == 0.20.0

## Usage
The hyper-parameter search range and optimal settings have been clearly stated in the codes (see the parser function in src/main.py).
* Train and Test

```
python main.py 
```


## Dataset

We provide three processed datasets: Book-Crossing, MovieLens-1M, and Last.FM.

We follow the paper " [Ripplenet: Propagating user preferences on the knowledge
graph for recommender systems](https://github.com/hwwang55/RippleNet)" to process data.


|                       |               | Book-Crossing | MovieLens-1M | Last.FM |
| :-------------------: | :------------ | ----------:   | --------: | ---------: |
| User-Item Interaction | #Users        |      17,860   |    6,036  |      1,872 |
|                       | #Items        |      14,967   |    2,445  |      3,846 |
|                       | #Interactions |     139,746   |  753,772  |      42,346|
|    Knowledge Graph    | #Entities     |      77,903   |    182,011|      9,366 |
|                       | #Relations    |          25   |         12|         60 |
|                       | #Triplets     |   151,500     |  1,241,996|     15,518 |


## Citation

If you want to use our codes in your research, please cite:
```
@inproceedings{KGIC2022,
  title     = {Improving Knowledge-aware Recommendation with Multi-level Interactive Contrastive Learning},
  author    = {
               Zou, Ding and 
               Wei, Wei and 
               Wang, Ziyang and
               Mao, Xian-Ling and
               Zhu, Feida and 
               Fang, Rui and 
               Chen, Dangyang},
  booktitle = {CIKM},
  year      = {2022}
}
```
