# GCORN - Bounding the Expected Robustness of Graph Neural Networks Subject to Node Feature Attacks

This repository is the official implementation of our paper "Bounding the Expected Robustness of Graph Neural Networks Subject to Node Feature Attacks"

The repository is divided into two folders. The first folder contains the "Empirical evaluation" based on existing attacks. The second folder contains our proposed probabilistic evaluation.

## Requirements

Code is written in Python 3.6 and requires:

- PyTorch
- Torch Geometric
- NetworkX


## Datasets
For node classification, the used datasets are as follows:
- Cora
- CiteSeer
- PubMed
- CS
- CoraML

All these datasets are part of the torch_geometric datasets and are directly downloaded when running the code.

For the OGBN-Arxiv, please refer to the relevant documentation. In our code, the dataset is downloaded directly if not found in the existing path.

## Training and Evaluation

To train and evaluate the model in the paper, the user should specify the following :

- Dataset : The dataset to be used
- hidden_dimension: The hidden dimension used in the model
- learning rate and epochs
- Budget: The budget of the attack
- Type of the attack: we evaluate using Random Attack and the PGD attack.

To run a normal code of GCORN without attack for the default values with Cora dataset:

```bash
python run_gcorn.py --dataset Cora
```

## Results reproduction
To reproduce the results in the paper that compare the GCN, RGCN and the GCORN, use the following:

```bash
python main_attack.py --dataset Cora --budget 0.5 --attack random
```


## OGBN-Arxiv

To reproduce the results for the OGBN dataset, a sparser version of the GCN, the RGCN and GCORN were made available for the users. Please refer to the folder "Code_ogb".
```bash
cd Code_ogb
python main_attack.py --budget 0.5
```
## For structural attacks

To use our code related to structure attack, the user should first download the DeepRobust package ( https://github.com/DSE-MSU/DeepRobust). Since we are using the GNNGUard as a baseline, we are using their official code from their github (https://github.com/mims-harvard/GNNGuard/tree/master).

As explained in the GNNGuard's original code, please substitute the files in the folder "deeprobust/graph/defense" by the one provided in our implementation. The file "gcorn.py" contains our proposed framework.

To reproduce the results for the Cora dataset, using the Mettack for instance with a 10% budget, you can use the following code:

```bash
cd Structural_perturbations
python attack_mettack.py --dataset cora --ptb_rate 0.1
```


## Reproduction
For other benchmarks used in our paper (AIRGNN, GCN-k), please refer directly to their available Github repository.
For all the details related to the code and the implementation, please refer to our paper.
