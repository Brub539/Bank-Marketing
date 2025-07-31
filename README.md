# Bank Marketing Classification Project

This project aims to build a machine learning model to predict whether a client will subscribe to a term deposit based on data from a Portuguese bank's telemarketing campaigns.

## Project Goal

The primary goal is to develop a high-performing classification model using LightGBM/XGBoost and compare its performance against simpler baseline models like Logistic Regression and Random Forest. The project emphasizes a full end-to-end workflow including data preprocessing, model training, evaluation, and interpretation.

## Directory Structure

bank-marketing-project/
├── data/ # Raw and Processed Data
├── models/ # Trained ML Models
├── notebooks/ # Jupyter notebooks for exploration
├── results/ # Model reports and visualizations
├── src/ # Source code for the pipeline
├── .gitignore # Files to be ignored by Git
└── README.md # Project overview


## How to Run

1.  **Setup and Preprocessing:** Run the main pipeline to download and prepare the data.
    ```bash
    python run_pipeline.py
    ```

2.  **Train Models:** Run the training scripts located in the `src/` directory.
    ```bash
    # Train Logistic Regression baseline
    python src/train_baseline.py

    # Train Random Forest baseline
    python src/train_random_forest.py

    # Train LightGBM model (coming soon)
    # python src/train_lightgbm.py
    ```