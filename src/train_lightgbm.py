# --- 1. Import Libraries ---
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import os
import sys
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

# --- 2. Setup Paths and Directories ---
MODEL_NAME = 'lightgbm_rfe_smote_target_encoding'
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
    print("Error: Target encoded data not found. Please run the correct pipeline first.")
    sys.exit()

# --- 4. Load Best Hyperparameters from File ---
# <<< THIS IS THE KEY CHANGE >>>
# Instead of a hardcoded dictionary, we load the results from our tuning notebook.
best_params_file = 'models/best_params/best_lgbm_params.joblib'
try:
    print(f"Loading best hyperparameters from '{best_params_file}'...")
    best_params = joblib.load(best_params_file)
    # The 'class_weight' parameter was for tuning; remove it for the SMOTE pipeline.
    if 'class_weight' in best_params:
        del best_params['class_weight']
    best_params['n_estimators'] = 1000 # Ensure we have enough estimators
    print("Hyperparameters loaded successfully.")
except FileNotFoundError:
    print(f"Error: Hyperparameter file not found at '{best_params_file}'.")
    print("Please run the '01_LGBM_Hyperparameter_Tuning.ipynb' notebook first.")
    sys.exit()

# --- 5. Define and Train using a SMOTE and RFE Pipeline ---
print("\nTraining LightGBM model with RFE and SMOTE...")
base_estimator = lgb.LGBMClassifier(random_state=42)
ultimate_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rfe', RFE(estimator=base_estimator, n_features_to_select=25, step=0.1)),
    ('classifier', lgb.LGBMClassifier(**best_params, random_state=42))
])
ultimate_pipeline.fit(X_train, y_train)

# --- 6, 7, 8: Evaluation and Saving (Code remains unchanged) ---
print("\nEvaluating the model...")
y_pred = ultimate_pipeline.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['No (deposit)', 'Yes (deposit)'])
print("\nClassification Report:")
print(report)
with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(report)
print(f"\nClassification report saved to '{RESULTS_DIR}/classification_report.txt'")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_estimator(ultimate_pipeline, X_test, y_test, ax=ax1, cmap='Purples')
ax1.set_title("Confusion Matrix - LGBM w/ RFE")
RocCurveDisplay.from_estimator(ultimate_pipeline, X_test, y_test, ax=ax2)
ax2.set_title("ROC Curve - LGBM w/ RFE")
plt.savefig(f'{RESULTS_DIR}/evaluation_plots.png')
plt.close('all')

print("\nSaving the final model...")
joblib.dump(ultimate_pipeline, f'{MODELS_DIR}/model.joblib')
print(f"Final model saved to '{MODELS_DIR}/model.joblib'")

print("\nUltimate LightGBM modeling complete!")