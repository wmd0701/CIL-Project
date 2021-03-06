# CIL-Project
This is the repo for Computational Intelligence Lab 2021 project, ETHZ. 

The task is **Collaborative Filtering** on movie ratings. There is a Kaggle competition for this course project on [Kaggle](https://www.kaggle.com/c/cil-collaborative-filtering-2021/). Our roup name is CIL Project.

The models are implemented using Pytorch. All notebook should run on Google Colab seamlessly. Data are automatically downloaded when running the notebook. However you may replace the kaggle account and api key in the notebook for the sake of privacy.


## Visualizations

### Movie histogram
<img src="https://user-images.githubusercontent.com/34072813/150688904-28c6931f-8ef8-4c0f-be10-8e3368ac6cef.png" width=40% height=40%>

### Rating histogram
<img src="https://user-images.githubusercontent.com/34072813/150688921-d1cb4552-669d-46e9-86ba-03b7b43ca03b.png" width=40% height=40%>

### Training MSE
<img src="https://user-images.githubusercontent.com/34072813/150688934-c5bc8e7c-b4db-48ef-87f4-6003e5f0403c.png" width=40% height=40%>

### Validation MSE
<img src="https://user-images.githubusercontent.com/34072813/150688942-af0d3e6b-2ad0-4358-8115-f4e0e7de7993.png" width=40% height=40%>

### Validation accuracy
<img src="https://user-images.githubusercontent.com/34072813/150688958-3bdbb309-71b0-49a0-bd75-e02fabfc396d.png" width=40% height=40%>



## Models

In total we implement 7 models. They are:

```
1. SVD++ with with SVD-based embedding initializaion
2. scaled sigmoided SVD++
3. item-based classification correction

4. Baseline - item-based KNN
5. Baseline - CNF
6. Baseline - SVD++ with uniformly initialized embeddings

7. Ensemble - bagging on scaled sigmoided SVD++ (model 2)
```

Model 1 and 6 are implemented in the same notebook (SVD++.ipynb). All other models are implemented stand-alone. Model 4, 5, 6 are baseline models. Model 7 is ensemble bagging over model 2.

## How to run model

Just open the notebook and run cells step by step. You can do this locally or on Google Colab. To accelerate the training process, you may switch to GPU session on Colab.

Especially, SVD++ can be run with two different strategies for embedding initialization (model 1 and 6). You can control initialization strategy by setting the variable ***svd_init*** to True/False. You can also opt for normal training style or ALS(alternating least square) training style by setting the variable ***ALS_train*** to True/False.

## Tips
1. Set the variable ***validate*** to True/False to control whether training with 90% data + validation with 10% data, or training with 100% data.

2. Please make sure you have enough RAM when running ensemble method (model 7). Otherwise, decrease ***n_jobs*** correspondingly. It may take hours before training is finished.

3. plot.ipynb is the notebook we used to produce plots for model comparison. All data are already gathered.
