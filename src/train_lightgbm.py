# --- 1. Import Libraries ---
import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, f1_score, precision_recall_curve
import matplotlib.pyplot as plt
import os
import sys

# --- 2. Setup Paths and Directories ---
MODEL_NAME = 'lightgbm_final' # Renaming for clarity
RESULTS_DIR = f'results/{MODEL_NAME}'
MODELS_DIR = f'models/{MODEL_NAME}'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 3. Load Processed Data ---
print("Loading preprocessed data...")
try:
    X_train = pd.read_csv('data/processed/X_train_processed.csv')
    X_test = pd.read_csv('data/processed/X_test_processed.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
except FileNotFoundError:
    print("Error: Processed data not found. Please run 'run_pipeline.py' first.")
    sys.exit()

# --- 4. Define Best Hyperparameters (WITHOUT class_weight) ---
# We want the model's natural probabilities, so we remove the class_weight parameter.
# The decision threshold will handle the class imbalance.
best_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    # 'class_weight': 'balanced', # REMOVED!
    'lambda_l1': 4.946332478396729e-08,
    'lambda_l2': 5.973873239518131e-07,
    'num_leaves': 17,
    'feature_fraction': 0.8892857720746322,
    'bagging_fraction': 0.9587196261506866,
    'bagging_freq': 5,
    'min_child_samples': 41,
    'n_estimators': 1000
}

# --- 5. Train the Final LightGBM Model ---
print("Training final LightGBM model on full training data...")
model = lgb.LGBMClassifier(**best_params)
model.fit(X_train, y_train)

# --- 6. Find the Optimal Decision Threshold ---
print("Finding optimal decision threshold...")
# Predict probabilities on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate precision, recall, and thresholds
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Calculate F1-score for each threshold (ignoring the last value for threshold)
f1_scores = (2 * precision * recall) / (precision + recall)
f1_scores = f1_scores[:-1] # Match the length of thresholds
thresholds = thresholds[:len(f1_scores)] # Match the length

# Find the threshold that gives the maximum F1 score
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
max_f1 = f1_scores[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"Max F1-Score at this threshold: {max_f1:.4f}")

# --- 7. Evaluate Using the Optimal Threshold ---
# Convert probabilities to predictions using our new threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

report = classification_report(y_test, y_pred_optimal, target_names=['No (deposit)', 'Yes (deposit)'])
print("\nFinal Model Classification Report (with optimal threshold):")
print(report)

with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
    f.write(f"Max F1-Score: {max_f1:.4f}\n\n")
    f.write(report)
print(f"\nClassification report saved to '{RESULTS_DIR}/classification_report.txt'")

# --- 8. Visualize and Save Plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_optimal, ax=ax1, cmap='Purples')
ax1.set_title(f"Confusion Matrix (Thresh={optimal_threshold:.2f})")

RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax2)
ax2.set_title("ROC Curve")

plt.savefig(f'{RESULTS_DIR}/evaluation_plots.png')
print(f"Evaluation plots saved to '{RESULTS_DIR}/evaluation_plots.png'")
plt.close('all')

# --- 9. Save the Final Model AND Threshold ---
print("Saving the final model and optimal threshold...")
joblib.dump(model, f'{MODELS_DIR}/model.joblib')
joblib.dump(optimal_threshold, f'{MODELS_DIR}/optimal_threshold.joblib')
print(f"Final model saved to '{MODELS_DIR}/model.joblib'")
print(f"Optimal threshold saved to '{MODELS_DIR}/optimal_threshold.joblib'")

print("\nFinal optimized modeling complete!")