# --- 1. Import Libraries ---
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
from imblearn.over_sampling import SMOTE
import joblib

# --- Helper Function for Logging ---
def log_step(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "="*70)
    print(f"{timestamp} - {message}")
    print("="*70)

# --- 2. Setup Paths ---
MODEL_NAME = 'deep_learning_mlp_tuned' # New name for our tuned model
RESULTS_DIR = f'results/{MODEL_NAME}'
MODELS_DIR = f'models/{MODEL_NAME}'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 3. Load Processed Data ---
log_step("STEP 1: Loading Preprocessed Data (Target Encoded)")
data_dir = 'data/processed_target_encoding'
try:
    X_train = pd.read_csv(f'{data_dir}/X_train_processed.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test_processed.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_test.csv').values.ravel()
    print("Data loaded successfully.")
except FileNotFoundError:
    sys.exit("Error: Data not found. Please run the pipeline script first.")

# --- 4. Apply SMOTE ---
log_step("STEP 2: Applying SMOTE for Class Imbalance")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"Resampled training shape: {X_train_res.shape}")

# --- 5. Load Best Hyperparameters from Tuning Script ---
log_step("STEP 3: Loading Tuned Architecture Blueprint")
best_params_file = 'models/best_params/best_dl_params.joblib'
try:
    best_params = joblib.load(best_params_file)
    print("Blueprint loaded successfully:")
    print(best_params)
except FileNotFoundError:
    sys.exit(f"FATAL: Blueprint file '{best_params_file}' not found. Please run 'src/tune/tune_deep_learning.py' first.")

# --- 6. Build the Tuned Model ---
log_step("STEP 4: Building Tuned MLP Model from Blueprint")
n_features = X_train_res.shape[1]
model = keras.Sequential()
model.add(layers.Input(shape=(n_features,)))

for i in range(best_params['n_layers']):
    n_units = best_params[f'n_units_l{i}']
    dropout_rate = best_params[f'dropout_l{i}']
    
    model.add(layers.Dense(n_units))
    model.add(layers.BatchNormalization()) # TIER 1 Improvement
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(dropout_rate))

model.add(layers.Dense(1, activation='sigmoid'))
optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
model.summary()

# --- 7. Define Callbacks for Smarter Training (TIER 1 Improvement) ---
log_step("STEP 5: Defining Training Callbacks")
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
print("Callbacks defined: EarlyStopping and ReduceLROnPlateau")

# --- 8. Train the Final Model ---
log_step("STEP 6: Training the Final Tuned MLP Model")
history = model.fit(
    X_train_res, y_train_res,
    epochs=200, # Train for a long time; early stopping will handle it
    batch_size=256,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# --- 9, 10, 11, 12: Evaluation and Saving (Boilerplate) ---
log_step("STEP 7: Generating and Saving Training History Plot")
pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.grid(True)
plt.title("Tuned MLP Model Training History")
plt.savefig(f'{RESULTS_DIR}/training_history.png')

log_step("STEP 8: Evaluating Model and Saving Final Report")
y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)
report = classification_report(y_test, y_pred, target_names=['No (deposit)', 'Yes (deposit)'])
print("\n--- Final Classification Report on Test Set ---")
print(report)
with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(report)

log_step("STEP 9: Generating and Saving Evaluation Plots")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax1, cmap='Reds')
ax1.set_title("Confusion Matrix - Tuned MLP")
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax2)
ax2.set_title("ROC Curve - Tuned MLP")
plt.savefig(f'{RESULTS_DIR}/evaluation_plots.png')
plt.close('all')

log_step("STEP 10: Saving the Trained Model")
model.save(f'{MODELS_DIR}/model.keras')
print(f"Model saved successfully to '{MODELS_DIR}/model.keras'")
print("\nTuned Deep Learning modeling complete!")