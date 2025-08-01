# --- 1. Import Libraries ---
import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt

# Model Imports
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import catboost as cbt

# Pipeline and Imbalance Handling
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Evaluation
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay

# --- 2. Setup Paths ---
MODEL_NAME = 'stacking_ensemble_tuned' # Name reflects that base models are tuned
RESULTS_DIR = f'results/{MODEL_NAME}'
MODELS_DIR = f'models/{MODEL_NAME}'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 3. Load Processed Data ---
print("Loading preprocessed data (TARGET ENCODED)...")
data_dir = 'data/processed_target_encoding'
try:
    X_train = pd.read_csv(f'{data_dir}/X_train_processed.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test_processed.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
except FileNotFoundError:
    sys.exit("Error: Target encoded data not found. Please run the pipeline script first.")

# --- 4. Load Tuned Hyperparameters for Base Models ---
print("Loading best hyperparameters for base models...")
try:
    lgbm_params = joblib.load('models/best_params/best_lgbm_params.joblib')
    rf_params = joblib.load('models/best_params/best_rf_params.joblib')
    
    # Remove class_weight as SMOTE will handle balancing in the pipeline
    if 'class_weight' in lgbm_params: del lgbm_params['class_weight']
    if 'class_weight' in rf_params: del rf_params['class_weight']
    
    print("Hyperparameters loaded successfully.")
except FileNotFoundError:
    sys.exit("FATAL: Tuned parameter files not found. Please run all tuning scripts (LGBM, RF) first.")

# --- 5. Define Tuned Base and Meta Models ---
print("Defining a diverse set of TUNED base models for stacking...")
estimators = [
    ('rf', RandomForestClassifier(**rf_params, random_state=42)),
    ('cat', cbt.CatBoostClassifier(verbose=0, random_state=42)),
    ('lgbm', lgb.LGBMClassifier(**lgbm_params, random_state=42))
]

# Use our best algorithm (a tuned LGBM) as the meta-model (the "manager")
# It will learn to combine the predictions from the base models.
meta_model = lgb.LGBMClassifier(**lgbm_params, random_state=42)
print("Using a tuned LightGBM as the meta-model.")

# Create the Stacking Classifier
# cv=5 means base model predictions are generated via 5-fold cross-validation (robust)
# n_jobs=-1 will use all available CPU cores to speed up training
stacking_clf = StackingClassifier(
    estimators=estimators, 
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

# --- 6. Create and Train the SMOTE Pipeline ---
print("Training Stacking Ensemble model with SMOTE... (This will take a significant amount of time)")
smote_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42, n_jobs=-1)),
    ('classifier', stacking_clf)
])

# This step is computationally very expensive.
smote_pipeline.fit(X_train, y_train)

# --- 7. Evaluate the Model ---
print("Evaluating the model...")
y_pred = smote_pipeline.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['No (deposit)', 'Yes (deposit)'])
print("\nClassification Report:")
print(report)
with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(report)

# --- 8. Visualize and Save Plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_estimator(smote_pipeline, X_test, y_test, ax=ax1, cmap='viridis')
ax1.set_title("Confusion Matrix - Tuned Stacking Ensemble")
RocCurveDisplay.from_estimator(smote_pipeline, X_test, y_test, ax=ax2)
ax2.set_title("ROC Curve - Tuned Stacking Ensemble")
plt.savefig(f'{RESULTS_DIR}/evaluation_plots.png')
plt.close('all')

# --- 9. Save the Model ---
joblib.dump(smote_pipeline, f'{MODELS_DIR}/model.joblib')
print(f"\nModel saved to '{MODELS_DIR}/model.joblib'")
print("\nTuned Stacking Ensemble modeling complete!")