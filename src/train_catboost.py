# --- 1. Import Libraries ---
import pandas as pd
import joblib
import os
import sys

# Model Imports
from catboost import CatBoostClassifier

# Pipeline and Imbalance Handling
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Evaluation Imports
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

# --- 2. Setup Paths and Directories ---
MODEL_NAME = 'catboost_target_encoding'
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
    print("Error: Target encoded data not found.")
    print("Please run 'run_pipeline_target_encoding.py' first.")
    sys.exit()

# --- 4. Define and Train the CatBoost Model using a SMOTE Pipeline ---
print("Training CatBoost model with SMOTE...")

# Create a pipeline that first applies SMOTE and then trains the CatBoost classifier.
smote_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    # CatBoost works well with default parameters, but we can set a few key ones.
    # verbose=0 prevents it from printing output during training.
    ('classifier', CatBoostClassifier(random_state=42, verbose=0, iterations=500))
])

# Fit the entire pipeline on the training data.
smote_pipeline.fit(X_train, y_train)

# --- 5. Evaluate the Model ---
print("Evaluating the CatBoost model...")
y_pred = smote_pipeline.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['No (deposit)', 'Yes (deposit)'])
print("\nClassification Report:")
print(report)

# Save the report
with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(report)
print(f"\nClassification report saved to '{RESULTS_DIR}/classification_report.txt'")

# --- 6. Visualize and Save Plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_estimator(smote_pipeline, X_test, y_test, ax=ax1, cmap='PuBu')
ax1.set_title("Confusion Matrix - CatBoost")

RocCurveDisplay.from_estimator(smote_pipeline, X_test, y_test, ax=ax2)
ax2.set_title("ROC Curve - CatBoost")

plt.savefig(f'{RESULTS_DIR}/evaluation_plots.png')
print(f"Evaluation plots saved to '{RESULTS_DIR}/evaluation_plots.png'")
plt.close('all')

# --- 7. Save the Trained Model ---
print("Saving the trained CatBoost model...")
joblib.dump(smote_pipeline, f'{MODELS_DIR}/model.joblib')
print(f"Model saved to '{MODELS_DIR}/model.joblib'")

print("\nCatBoost modeling complete!")