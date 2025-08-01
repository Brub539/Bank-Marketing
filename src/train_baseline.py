# --- 1. Import Libraries ---
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import os
import sys
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# --- 2. Setup Paths and Directories ---
MODEL_NAME = 'logistic_regression_smote_te' # TE = Target Encoding
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
    print(f"Error: Data not found in '{data_dir}'. Please run the pipeline script first.")
    sys.exit()

# --- 4. Define and Train Logistic Regression Model using a SMOTE Pipeline ---
print("Training Logistic Regression baseline model with SMOTE...")
smote_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'))
])
smote_pipeline.fit(X_train, y_train)

# --- 5. Evaluate the Model ---
print("Evaluating the baseline model...")
y_pred = smote_pipeline.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['No (deposit)', 'Yes (deposit)'])
print("\nClassification Report:")
print(report)
with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(report)
print(f"\nClassification report saved to '{RESULTS_DIR}/classification_report.txt'")

# --- 6. Visualize and Save Plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_estimator(smote_pipeline, X_test, y_test, ax=ax1, cmap='Blues')
ax1.set_title("Confusion Matrix - Logistic Regression")
RocCurveDisplay.from_estimator(smote_pipeline, X_test, y_test, ax=ax2)
ax2.set_title("ROC Curve - Logistic Regression")
plt.savefig(f'{RESULTS_DIR}/evaluation_plots.png')
plt.close('all')

# --- 7. Save the Trained Model ---
print("Saving the trained model...")
joblib.dump(smote_pipeline, f'{MODELS_DIR}/model.joblib')
print(f"Model saved to '{MODELS_DIR}/model.joblib'")
print("\nBaseline modeling complete!")