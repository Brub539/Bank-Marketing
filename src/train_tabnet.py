# --- 1. Import Libraries ---
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import joblib

# Model and Data Handling Imports
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay

# --- Helper Function for Logging ---
def log_step(message):
    """Prints a formatted log message with a timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "="*70)
    print(f"{timestamp} - {message}")
    print("="*70)

# --- 2. Setup Paths and Directories ---
MODEL_NAME = 'tabnet_smote_target_encoding'
RESULTS_DIR = f'results/{MODEL_NAME}'
MODELS_DIR = f'models/{MODEL_NAME}'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 3. Load Processed Data ---
log_step("STEP 1: Loading Preprocessed Data")
try:
    data_dir = 'data/processed_target_encoding'
    X_train = pd.read_csv(f'{data_dir}/X_train_processed.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test_processed.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    print(f"Data loaded successfully from '{data_dir}'.")
except FileNotFoundError:
    print(f"Error: Processed data not found in '{data_dir}'.")
    print("Please ensure the correct pipeline has been run first.")
    sys.exit()

# --- 4. Apply SMOTE to the Training Data ---
log_step("STEP 2: Applying SMOTE for Class Imbalance")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train.to_numpy(), y_train)
print(f"Original training shape: {X_train.shape} | Resampled training shape: {X_train_res.shape}")
X_test_np = X_test.to_numpy()

# --- 5. Define and Train the TabNet Model ---
log_step("STEP 3: Training the TabNet Model")
model = TabNetClassifier(verbose=1, seed=42)

print("Starting model training...")
# --- CHANGE APPLIED: Removed eval_set for a more stable training run with SMOTE ---
# Patience will now monitor the training loss for convergence.
model.fit(
    X_train=X_train_res,
    y_train=y_train_res,
    patience=20,
    max_epochs=100
)

# --- 6. Plot and Save Training History ---
log_step("STEP 4: Generating and Saving Training History Plot")
# --- CHANGE APPLIED: Access the 'loss' key directly from the history dict ---
# This is a robust way to get the training loss regardless of early stopping.
history_df = pd.DataFrame({'loss': model.history['loss']})
history_df.plot(figsize=(10, 7))
plt.grid(True)
plt.title("TabNet Model Training Loss History")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(f'{RESULTS_DIR}/training_history.png')
print(f"Training history plot saved to '{RESULTS_DIR}/training_history.png'")

# --- 7. Evaluate and Save Report ---
log_step("STEP 5: Evaluating Model and Saving Final Report")
y_pred_proba = model.predict_proba(X_test_np)[:, 1]
y_pred = model.predict(X_test_np)
report = classification_report(y_test, y_pred, target_names=['No (deposit)', 'Yes (deposit)'])
print("\n--- Final Classification Report on Test Set ---")
print(report)
with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(report)
print(f"\nClassification report saved to '{RESULTS_DIR}/classification_report.txt'")

# --- 8. Visualize and Save Evaluation Plots ---
log_step("STEP 6: Generating and Saving Evaluation Plots")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax1, cmap='plasma')
ax1.set_title("Confusion Matrix - TabNet w/ SMOTE")
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax2)
ax2.set_title("ROC Curve - TabNet w/ SMOTE")
plt.savefig(f'{RESULTS_DIR}/evaluation_plots.png')
plt.close('all')

# --- 9. Save the Trained Model ---
log_step("STEP 7: Saving the Trained Model")
joblib.dump(model, f'{MODELS_DIR}/model.joblib')
print(f"Model saved successfully to '{MODELS_DIR}/model.joblib'")
print("\nTabNet modeling complete!")