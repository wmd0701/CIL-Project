# CIL-Project
This is the repo for Computational Intelligence Lab 2020 project, ETHZ. Kaggle group name: CIL Project.

The models are implemented using Pytorch. All notebook should run on Google Colab seamlessly. 

## Models

We implement in total 6 models, they are:

```
Baseline - item-based KNN
Baseline - CNF
Baseline - SVD++ with uniformly initialized embeddings

SVD++ with with SVD-based embedding initializaion
scaled sigmoided SVD++
item-based classification correction
```

## How to predict with the final model

1. Activate Enviroment
```
conda activate mp_project3
```
2. Make predictions with the last model. Look in the out directory for the last iteration number (should be 40000). Use that model to make predictions
```
cd codebase/ && bsub -n 6 -W 00:30 -o out.txt -R "rusage[mem=4096, ngpus_excl_p=1]" python test.py ../configs/modelfinal.yaml --model_file model_40000.pt
```
