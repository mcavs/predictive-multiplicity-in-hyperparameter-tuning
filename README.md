# Predictive Multiplicity in Hyperparameter Tuning: A Unified Investigation of Tabular Classification and Regression

This repository contains the replication code, experimental configurations, and analysis pipelines for the paper:
> **Predictive Multiplicity in Hyperparameter Tuning: A Unified Investigation of Tabular Classification and Regression**

---

## Hyperparameter Search Spaces and Defaults (Table A1)

The table below summarizes the hyperparameter search spaces, bound constraints, parameter types, and default configurations evaluated across the six learning algorithms for both binary classification and tabular regression tasks:

| Model | Hyperparameter | Type | Lower | Upper | Classification Default | Regression Default |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Elastic Net** () |  | numeric | zsh$ | $ | $ | $ |
| |  | numeric | ^{-10}$ | ^{10}$ | zsh$ | zsh$ |
| **Decision Tree** () |  | numeric | zsh$ | $ | zsh.1$ | zsh.1$ |
| |  | integer | $ | $ | $ | $ |
| |  | integer | $ | $ | $ | $ |
| |  | integer | $ | $ | $ | $ |
| **569XNN** () |  | integer | $ | $ | $ | $ |
| **SVM** () |  | numeric | ^{-10}$ | ^{10}$ | $ | $ |
| |  | numeric | ^{-10}$ | ^{10}$ | /p$ | /p$ |
| **Random Forest** () |  | integer | $ | $ | $ | $ |
| |  | numeric | zsh.1$ | $ | $ | $ |
| |  | integer | $ | $ | $\lfloor\sqrt{p}floor$ | $\lfloor p/3floor$ |
| |  | integer | $ | $ | $ | $ |
| **XGBoost** () |  | categorical | --- | --- |  |  |
| |  | integer | $ | $ | $ | $ |
| |  | numeric | zsh$ | $ | zsh.3$ | zsh.3$ |
| |  | numeric | zsh.1$ | $ | $ | $ |
| |  | integer | $ | $ | $ | $ |
| |  | numeric | $ | $ | $ | $ |
| |  | numeric | zsh.01$ | $ | $ | $ |
| |  | numeric | zsh.01$ | $ | $ | $ |
| |  | numeric | ^{-10}$ | ^{10}$ | $ | $ |
| |  | numeric | ^{-10}$ | ^{10}$ | zsh$ | zsh$ |

*$ denotes the number of features and $ denotes the number of observations. For Random Forest, the default  is $\max(1, \lfloor\sqrt{p}floor)$ in classification and $\max(1, \lfloor p/3floor)$ in regression, while  defaults to $ and $, respectively.*
