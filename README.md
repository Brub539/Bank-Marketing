Bank Marketing Predictive Modeling Project
1. Project Overview
This project undertakes a comprehensive, end-to-end machine learning workflow to predict the success of telemarketing campaigns for a Portuguese banking institution. The primary objective is to build and optimize a binary classification model that predicts whether a client will subscribe to a term deposit (y).
The project emphasizes a robust and ethical methodology, starting with a realistic, non-leaky dataset and progressively applying advanced feature engineering, imbalance handling, hyperparameter tuning, and a diverse suite of modeling techniques—from traditional baselines to state-of-the-art deep learning and ensemble methods.
This document details the data processing pipeline, the evolution of modeling strategies, a comparative analysis of all models, and the final conclusions.
2. The Data Pipeline: From Raw Data to Engineered Features
The core of this project is a robust, automated data pipeline (run_pipeline.py) built on best practices to transform the raw data into a feature-rich dataset ready for modeling.
Data Source: The project uses the bank-additional-full.csv dataset, which includes 20 input variables and is chronologically ordered from May 2008 to November 2010.
Validation Strategy: A time-based split is used to ensure a realistic evaluation. The model is trained on the first 80% of the data and tested on the most recent 20%, simulating a real-world deployment scenario where past data is used to predict future outcomes.
Advanced Feature Engineering: The pipeline applies several tiers of feature engineering to enrich the dataset:
Outlier Capping: Numeric features (age, campaign, etc.) have outliers capped using the IQR method to improve model stability.
Group-Based Aggregates: New features are created to provide crucial context, such as the mean age or std dev of euribor3m for a client's specific job or education level.
Categorical Encoding: High-cardinality features are encoded using Target Encoding, which replaces categories with their historical probability of success, packing more information into each feature.
Data Leakage Prevention: The duration column, a known data leak that is only available after a call is made, is handled in two separate experimental runs. This allows us to distinguish between a realistic predictive model and a purely academic benchmark.
3. Modeling and Optimization Strategy
The project systematically builds and evaluates a wide range of models to identify the optimal solution. The entire workflow is automated via the run_project.py script.
Baselines: A LogisticRegression model serves as the initial benchmark.
Core Models: RandomForest, LightGBM, and CatBoost are evaluated as powerful tree-based models.
Deep Learning: A standard MLP (Deep Learning) model and a specialized TabNet model are built to test modern neural network architectures on tabular data.
Class Imbalance: The significant class imbalance in the target variable is addressed using SMOTE (Synthetic Minority Over-sampling TEchnique), which creates synthetic data for the minority class to ensure the models can effectively learn its patterns.
Hyperparameter Tuning: Key models (LightGBM, Random Forest, MLP) are rigorously tuned using the Optuna framework to find their optimal architecture and regularization parameters, moving them from baseline performance to highly optimized contenders.
Ensembles: Advanced Voting and Stacking ensembles are constructed to test if combining the predictions of multiple diverse models can yield superior performance.
4. Results and Analysis
Two major experiments were conducted to provide a complete picture of the model's capabilities. The primary metric for success is the F1-Score for the positive ('Yes') class, as it provides the best balance between precision and recall for this imbalanced business problem.
Experiment 1: Realistic Predictive Model (No duration column)
This experiment simulates a real-world business problem where the goal is to predict which customers to call before the call is made. The duration column is correctly excluded.
Final Model Comparison (Realistic Scenario):
| Model | Precision (Yes) | Recall (Yes) | F1-Score (Yes) |
| :--- | :---: | :---: | :---: |
| Tabnet Smote Target Encoding | 0.41 | 0.61 | 0.49 |
| Logistic Regression Smote Te | 0.36 | 0.56 | 0.44 |
| Voting Ensemble Smote V2 | 0.59 | 0.19 | 0.29 |
| Lightgbm Rfe Smote Target Encoding | 0.57 | 0.18 | 0.27 |
| Stacking Ensemble Smote V2 | 0.57 | 0.09 | 0.15 |
| Catboost Target Encoding | 0.57 | 0.07 | 0.13 |
| Deep Learning Mlp Smote Te | 0.14 | 0.07 | 0.09 |
| Random Forest Target Encoding | 0.49 | 0.01 | 0.02 |
Conclusion (Realistic): The champion model is Tabnet Smote Target Encoding with an F1-Score of 0.49.
Analysis: In a realistic scenario with noisy, high-dimensional data, the specialized TabNet architecture excelled. Its internal attention mechanism allowed it to perform automatic feature selection, focusing on the most important signals and ignoring the noise that confused the powerful but less-specialized tree-based ensembles.
Experiment 2: Academic Benchmark Model (With duration column)
This experiment includes the leaky duration column to replicate the conditions often found in academic papers and online benchmarks, where the goal is post-call analysis.
Final Model Comparison (Benchmark Scenario):
| Model | Precision (Yes) | Recall (Yes) | F1-Score (Yes) |
| :--- | :---: | :---: | :---: |
| Rf Tuned Smote Optimal Thresh| 0.51 | 0.83 | 0.64 |
| Deep Learning Mlp Tuned | 0.51 | 0.67 | 0.58 |
| Logistic Regression Smote Te | 0.54 | 0.59 | 0.56 |
| Tabnet Smote Target Encoding | 0.55 | 0.34 | 0.42 |
| Rf Tuned Class Weight | 0.64 | 0.31 | 0.42 |
| Stacking Ensemble Smote V2 | 0.65 | 0.28 | 0.39 |
| Catboost Target Encoding | 0.69 | 0.27 | 0.38 |
| Stacking Ensemble Tuned | 0.57 | 0.21 | 0.31 |
| Lightgbm Rfe Smote Target Encoding| 0.69 | 0.16 | 0.26 |
Conclusion (Benchmark): The champion model is the Tuned Random Forest with SMOTE and an Optimized Threshold with a final F1-Score of 0.64.
Analysis: When a powerful, high-signal feature like duration is introduced, the problem shifts. A well-tuned traditional model like Random Forest was able to capitalize on this strong signal more effectively than any other model. The optimization of the decision threshold was also a critical factor, pushing the model's recall to an impressive 0.83.
5. Final Conclusion
This project successfully developed and evaluated a suite of machine learning models for a challenging, real-world marketing problem. It demonstrates the critical importance of a sound, ethical methodology that includes a time-based validation strategy and the exclusion of leaky features for a realistic predictive task.
The final recommendation would be to deploy the TabNet model (F1-Score: 0.49) for the business problem of pre-call targeting. For academic or post-call analysis, the Tuned Random Forest (F1-Score: 0.64) provides a powerful benchmark. The entire workflow, from data engineering to final analysis, is fully automated.