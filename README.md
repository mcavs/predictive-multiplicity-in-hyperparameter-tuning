# Predictive Multiplicity in Hyperparameter Tuning: A Unified Investigation of Tabular Classification and Regression

This repository contains the replication code, experimental configurations, and analysis pipelines for the paper:
> **Predictive Multiplicity in Hyperparameter Tuning: A Unified Investigation of Tabular Classification and Regression**

---

## Hyperparameter Search Spaces and Defaults (Table A1)

The table below summarizes the hyperparameter search spaces, bound constraints, parameter types, and default configurations evaluated across the six learning algorithms for both binary classification and tabular regression tasks:

| Model | Hyperparameter | Type | Lower | Upper | Classification Default | Regression Default |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Elastic Net** (`glmnet`) | `alpha` | numeric | 0 | 1 | 1 | 1 |
| | `lambda` | numeric | 2^(-10) | 2^(10) | 0 | 0 |
| **Decision Tree** (`rpart`) | `cp` | numeric | 0 | 1 | 0.1 | 0.1 |
| | `maxdepth` | integer | 1 | 30 | 30 | 30 |
| | `minbucket` | integer | 1 | 60 | 7 | 7 |
| | `minsplit` | integer | 1 | 60 | 20 | 20 |
| **k-NN** (`kknn`) | `k` | integer | 1 | 30 | 7 | 7 |
| **SVM** (`e1071`) | `cost` | numeric | 2^(-10) | 2^(10) | 1 | 1 |
| | `gamma` | numeric | 2^(-10) | 2^(10) | 1/p | 1/p |
| **Random Forest** (`ranger`) | `num.trees` | integer | 1 | 2000 | 500 | 500 |
| | `sample.fraction` | numeric | 0.1 | 1 | 1 | 1 |
| | `mtry` | integer | 1 | p | floor(sqrt(p)) | floor(p/3) |
| | `min.node.size` | integer | 1 | n | 1 | 5 |
| **XGBoost** (`xgboost`) | `booster` | categorical | --- | --- | `gbtree` | `gbtree` |
| | `nrounds` | integer | 1 | 5000 | 500 | 500 |
| | `eta` | numeric | 0 | 1 | 0.3 | 0.3 |
| | `subsample` | numeric | 0.1 | 1 | 1 | 1 |
| | `max_depth` | integer | 1 | 15 | 6 | 6 |
| | `min_child_weight` | numeric | 1 | 14 | 1 | 1 |
| | `colsample_bytree` | numeric | 0.01 | 1 | 1 | 1 |
| | `colsample_bylevel` | numeric | 0.01 | 1 | 1 | 1 |
| | `lambda` | numeric | 2^(-10) | 2^(10) | 1 | 1 |
| | `alpha` | numeric | 2^(-10) | 2^(10) | 0 | 0 |

*p denotes the number of features and n denotes the number of observations. For Random Forest, the default `mtry` is max(1, floor(sqrt(p))) in classification and max(1, floor(p/3)) in regression, while `min.node.size` defaults to 1 and 5, respectively.*
