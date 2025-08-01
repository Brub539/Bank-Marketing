# --- 1. Import Libraries ---
import pandas as pd
import joblib
import os
import sys
import matplotlib.pyplot as plt

# Model Imports
from sklearn.ensemble import StackingClassifier, RandomForestClassifier # Added RF
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import catboost as cbt # Added CatBoost

# Pipeline and Imbalance Handling
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Evaluation
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay

# --- 2. Setup Paths ---
MODEL_NAME = 'stacking_ensemble_smote_v2' # New version
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

# --- 4. Define Base and Meta Models ---
print("Defining a diverse set of base models for stacking...")
lgbm_params = {'objective': 'binary', 'metric': 'binary_logloss', 'n_estimators': 1000}

# --- ADDITION: Create a more powerful and diverse set of base models ---
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('cat', cbt.CatBoostClassifier(verbose=0, random_state=42)),
    ('lgbm', lgb.LGBMClassifier(**lgbm_params, random_state=42))
]

# The meta-model remains a Logistic Regression to learn the optimal combination
meta_model = LogisticRegression(max_iter=1000)
# --- END OF ADDITION ---

# Create the Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=estimators, 
    final_estimator=meta_model,
    cv=5 # This is computationally expensive but robust
)

# --- 5. Create and Train the SMOTE Pipeline ---
print("Training Stacking Ensemble model with SMOTE...")
smote_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', stacking_clf)
])
smote_pipeline.fit(X_train, y_train)

# --- 6, 7, 8: Evaluation and Saving (Code remains the same) ---
print("Evaluating the model...")
y_pred = smote_pipeline.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['No (deposit)', 'Yes (deposit)'])
print("\nClassification Report:")
print(report)
with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(report)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_estimator(smote_pipeline, X_test, y_test, ax=ax1, cmap='viridis')
ax1.set_title("Confusion Matrix - Stacking Ensemble")
RocCurveDisplay.from_estimator(smote_pipeline, X_test, y_test, ax=ax2)
ax2.set_title("ROC Curve - Stacking Ensemble")
plt.savefig(f'{RESULTS_DIR}/evaluation_plots.png')

joblib.dump(smote_pipeline, f'{MODELS_DIR}/model.joblib')
print(f"\nModel saved to '{MODELS_DIR}/model.joblib'")
print("\nStacking Ensemble modeling complete!")