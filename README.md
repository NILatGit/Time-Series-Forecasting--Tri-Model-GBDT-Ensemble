# Time-Series Forecasting: Tri-Model GBDT Ensemble

## Overview

This repository contains a robust machine learning pipeline designed for complex time-series forecasting on tabular data. The core implementation and architecture were developed and exported directly from a Kaggle competition environment.

For complete reproducibility, the entire original dataset has been included directly in this repository using the exact original file names. The scripts and notebooks can be executed out-of-the-box without any modifications to the data paths.

## Dataset Structure

The raw data files are preserved in the root directory:

* `orders_train.csv`: The historical time-series data used for model training and validation.
* `orders_test.csv`: The holdout temporal data used to generate the final predictions.
* `submission_final.csv`: The target format template for the final prediction output.

## Model Architecture

The predictive engine utilizes a weighted ensemble of three state-of-the-art Gradient Boosted Decision Tree (GBDT) frameworks. To effectively model temporal dependencies and combat the cross-validation-to-leaderboard disconnect, the pipeline employs an expanding-window time-series split.

* **LightGBM:** Utilized for its high execution speed and memory efficiency via histogram-based binning, providing a strong baseline for tabular feature interactions.
* **CatBoost:** Integrated specifically for its native, highly optimized handling of categorical variables without requiring manual target encoding.
* **XGBoost:** Deployed for its aggressive regularization parameters to prevent overfitting on noisy, non-stationary temporal data.

## Setup and Execution

To replicate the environment and generate predictions:

1. Clone this repository to your local machine.
2. Ensure you have the necessary dependencies installed (`pandas`, `numpy`, `lightgbm`, `catboost`, `xgboost`, `scikit-learn`).
3. Execute the primary notebook/script to run the preprocessing pipeline, train the tri-model ensemble, and output the final prediction file.
