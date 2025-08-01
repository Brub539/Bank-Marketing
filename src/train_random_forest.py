# --- 1. Import Libraries ---
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import os
import sys
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# --- 2. Load Processed Data ---
print("="*70)
print("Loading preprocessed data (TARGET ENCODED)...")
data_dir = 'data/processed_target_encoding'
try:
    X_train = pd.read_csv(f'{data_dir}/X_train_processed.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test_processed.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Data not found in '{data_dir}'. Please run the pipeline script first.")
    sys.exit()

# --- 3. Load Best Hyperparameters ---
best_params_file = 'models/best_params/best_rf_params.joblib'
try:
    print(f"Loading best hyperparameters from '{best_params_file}'...")
    best_params = joblib.load(best_params_file)
    print("Hyperparameters loaded successfully.")
except FileNotFoundError:
    print(f"FATAL: Hyperparameter file '{best_params_file}' not found.")
    print("Please run 'src/tune_random_forest.py' first.")
    sys.exit()

# ==============================================================================
# EXPERIMENT A: SMOTE with Optimized Decision Threshold
# ==============================================================================
print("\n" + "="*70)
print("RUNNING EXPERIMENT A: SMOTE with Optimized Threshold")
print("="*70)

# Setup Paths
MODEL_NAME_A = 'rf_tuned_smote_optimal_thresh'
RESULTS_DIR_A = f'results/{MODEL_NAME_A}'
MODELS_DIR_A = f'models/{MODEL_NAME_A}'
os.makedirs(RESULTS_DIR_A, exist_ok=True)
os.makedirs(MODELS_DIR_A, exist_ok=True)

# Define and Train the SMOTE pipeline
params_A = best_params.copy()
if 'class_weight' in params_A:
    del params_A['class_weight']

print("Training model with SMOTE...")
smote_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(**params_A))
])
smote_pipeline.fit(X_train, y_train)

# --- Find the Optimal Decision Threshold ---
print("Finding optimal decision threshold...")
y_pred_proba = smote_pipeline.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1_scores = (2 * precision * recall) / (precision + recall)
f1_scores = f1_scores[:-1]
thresholds = thresholds[:len(f1_scores)]
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
max_f1 = f1_scores[optimal_idx]

print(f"Optimal Threshold found: {optimal_threshold:.4f} (giving max F1-score: {max_f1:.4f})")

# Evaluate using the optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
report_A = classification_report(y_test, y_pred_optimal, target_names=['No (deposit)', 'Yes (deposit)'])

print("\nClassification Report (SMOTE w/ Optimal Threshold):")
print(report_A)
with open(f'{RESULTS_DIR_A}/classification_report.txt', 'w') as f:
    f.write(f"Optimal Threshold: {optimal_threshold:.4f}\nMax F1-Score: {max_f1:.4f}\n\n{report_A}")

# Visualize and Save
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_optimal, ax=ax1, cmap='Greens')
ax1.set_title(f"Confusion Matrix (Thresh={optimal_threshold:.2f})")
RocCurveDisplay.from_estimator(smote_pipeline, X_test, y_test, ax=ax2)
ax2.set_title("ROC Curve - Tuned RF w/ SMOTE")
plt.savefig(f'{RESULTS_DIR_A}/evaluation_plots.png')
plt.close('all')

# Save the model AND the threshold
joblib.dump(smote_pipeline, f'{MODELS_DIR_A}/model.joblib')
joblib.dump(optimal_threshold, f'{MODELS_DIR_A}/optimal_threshold.joblib')
print(f"Model and threshold for Experiment A saved to '{MODELS_DIR_A}/'")

# ==============================================================================
# EXPERIMENT B: Using class_weight='balanced_subsample' (No SMOTE)
# ==============================================================================
print("\n" + "="*70)
print("RUNNING EXPERIMENT B: Using class_weight='balanced_subsample'")
print("="*70)

# Setup Paths
MODEL_NAME_B = 'rf_tuned_class_weight'
RESULTS_DIR_B = f'results/{MODEL_NAME_B}'
MODELS_DIR_B = f'models/{MODEL_NAME_B}'
os.makedirs(RESULTS_DIR_B, exist_ok=True)
os.makedirs(MODELS_DIR_B, exist_ok=True)

# Define and Train the model
params_B = best_params.copy()
params_B['class_weight'] = 'balanced_subsample' # Add the class weight parameter

print("Training model with class_weight='balanced_subsample'...")
model_B = RandomForestClassifier(**params_B)
model_B.fit(X_train, y_train)

# Evaluate the model (using default 0.5 threshold)
y_pred_B = model_B.predict(X_test)
report_B = classification_report(y_test, y_pred_B, target_names=['No (deposit)', 'Yes (deposit)'])

print("\nClassification Report (class_weight='balanced_subsample'):")
print(report_B)
with open(f'{RESULTS_DIR_B}/classification_report.txt', 'w') as f:
    f.write(report_B)

# Visualize and Save
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_estimator(model_B, X_test, y_test, ax=ax1, cmap='Oranges')
ax1.set_title("Confusion Matrix - Tuned RF w/ class_weight")
RocCurveDisplay.from_estimator(model_B, X_test, y_test, ax=ax2)
ax2.set_title("ROC Curve - Tuned RF w/ class_weight")
plt.savefig(f'{RESULTS_DIR_B}/evaluation_plots.png')
plt.close('all')

# Save the model
joblib.dump(model_B, f'{MODELS_DIR_B}/model.joblib')
print(f"Model for Experiment B saved to '{MODELS_DIR_B}/'")

print("\n" + "="*70)
print("Both Random Forest optimization experiments are complete.")
print("="*70)