# multiTune: Multiplicity-Aware Hyperparameter Recommendation & Auditing Tool

An interactive R Shiny decision-support tool designed for researchers and practitioners who want to perform hyperparameter optimization without inducing severe predictive multiplicity (decision instability) on their tabular datasets.

## Key Features

1. **Multiplicity Risk Scoring:** Evaluates the inherent decision volatility of candidate algorithms (Random Forests, XGBoost, Decision Tree, SVM, Elastic Net) for both classification and regression tasks.
2. **Instant Safe Search Boundaries:** Computes calibrated, Pareto-optimal hyperparameter intervals based on meta-surrogate response surfaces, guaranteeing positive generalization gains while restricting discrepancy to the lower quartile.
3. **Bivariate Pareto Frontier Explorer:** Allows practitioners to dynamically tune the multiplicity penalty parameter $\beta \in [0, 1]$ to visualize the trade-off between validation gains and individual prediction stability.
4. **Domain Practitioner Advice:** Provides concrete algorithmic guidance for preventing subtree thrashing in single trees, leaf variance in random forests, and residual instability in gradient boosting.

## How to Run

In an R console with the required dependencies installed (`shiny`, `bslib`, `ggplot2`):

```R
shiny::runApp("shiny_app")
```

## Methodology

This tool implements the empirical findings and meta-surrogate formulation established in:
> Cavus, M. (2026). *Predictive Multiplicity in Hyperparameter Tuning: A Unified Investigation of Tabular Classification and Regression.*
