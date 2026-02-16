# Case Solution: First-Time Investor Prediction

This repository contains a proposed solution to a case with the goal of identifying the **1,000 customers most likely to invest for the first time**.

The work is exploratory and iterative: notebooks include data analysis, feature engineering, and multiple modeling approaches used to evaluate different ways of solving the problem.

## Repository Overview

### `Exploration.ipynb`
Initial data exploration of structure, distributions, and quality.

- Investigates dataset content and key variables
- Identifies irregularities/anomalies
- Documents potential handling strategies

### `Segmentation.ipynb`
Customer clustering and segmentation analysis.

- Explores whether customer groups show useful behavioral patterns
- Compares groups with investors vs non-investors
- Looks for patterns among recent first-time investors

### `Feature engineering.ipynb`
Feature creation and transformation for downstream modeling.

- Builds predictive candidate features
- Prepares engineered inputs for classification and time-to-event models

### `model.ipynb`
Main modeling notebook with three different approaches:

1. **Logistic Regression baseline**
2. **XGBoost classifier**
3. **Weibull RNN time-to-event model**

This notebook also includes code to generate the final ranked list of the **top 1,000 potential first-time investors** for each approach.

## Notes

- The repository is designed as a case solution and exploration workflow.
- Some parts are intentionally experimental to compare alternative methods.