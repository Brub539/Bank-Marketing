# --- 1. Import Libraries ---
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import joblib

# --- Helper Function for Logging ---
def log_step(message):
    """Prints a formatted log message with a timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "="*70)
    print(f"{timestamp} - {message}")
    print("="*70)

# --- 2. Setup Paths and Directories ---
MODEL_NAME = 'deep_learning_mlp_smote_te' # TE = Target Encoding
RESULTS_DIR = f'results/{MODEL_NAME}'
MODELS_DIR = f'models/{MODEL_NAME}'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 3. Load Processed Data ---
log_step("STEP 1: Loading Preprocessed Data")
try:
    # --- CRITICAL: Ensure we use the best data ---
    data_dir = 'data/processed_target_encoding'
    X_train = pd.read_csv(f'{data_dir}/X_train_processed.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test_processed.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    print(f"Data loaded successfully from '{data_dir}'.")
except FileNotFoundError:
    print(f"Error: Data not found in '{data_dir}'. Please run the pipeline script first.")
    sys.exit()

# --- 4. Apply SMOTE to the Training Data ---
log_step("STEP 2: Applying SMOTE for Class Imbalance")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"Original training shape: {X_train.shape} | Resampled training shape: {X_train_res.shape}")

# --- 5. Define, Compile, and Summarize Model ---
log_step("STEP 3: Defining and Compiling the MLP Model")
n_features = X_train_res.shape[1]
model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(128, activation='relu', name='Hidden_Layer_1'),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu', name='Hidden_Layer_2'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid', name='Output_Layer')
], name="MLP_Model_TE")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
model.summary()

# --- 6. Train the Model ---
log_step("STEP 4: Training the MLP Model")
history = model.fit(
    X_train_res, y_train_res,
    epochs=50, batch_size=256,
    validation_data=(X_test, y_test),
    verbose=1
)

# --- 7. Plot Training History ---
log_step("STEP 5: Generating and Saving Training History Plot")
pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("MLP Model Training History")
plt.savefig(f'{RESULTS_DIR}/training_history.png')
print(f"Training history plot saved to '{RESULTS_DIR}/training_history.png'")

# --- 8. Evaluate and Save Report ---
log_step("STEP 6: Evaluating Model and Saving Final Report")
y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)
report = classification_report(y_test, y_pred, target_names=['No (deposit)', 'Yes (deposit)'])
print("\n--- Final Classification Report on Test Set ---")
print(report)
with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(report)
print(f"\nClassification report saved to '{RESULTS_DIR}/classification_report.txt'")

# --- 9. Visualize and Save Evaluation Plots ---
log_step("STEP 7: Generating and Saving Evaluation Plots")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax1, cmap='Reds')
ax1.set_title("Confusion Matrix - MLP")
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax2)
ax2.set_title("ROC Curve - MLP")
plt.savefig(f'{RESULTS_DIR}/evaluation_plots.png')
plt.close('all')

# --- 10. Save the Trained Model ---
log_step("STEP 8: Saving the Trained Model")
model.save(f'{MODELS_DIR}/model.keras')
print(f"Model saved successfully to '{MODELS_DIR}/model.keras'")
print("\nDeep Learning modeling complete!")