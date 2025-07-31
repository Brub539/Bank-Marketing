# --- 1. Import Libraries ---
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay

# --- 2. Setup Paths and Directories ---
MODEL_NAME = 'deep_learning_mlp'
RESULTS_DIR = f'results/{MODEL_NAME}'
MODELS_DIR = f'models/{MODEL_NAME}'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Check for Graphviz
try:
    from tensorflow.keras.utils import plot_model
    CAN_PLOT_MODEL = True
except ImportError:
    print("Warning: 'pydot' or 'graphviz' not found. Cannot plot model architecture.")
    CAN_PLOT_MODEL = False

# --- 3. Load Processed Data ---
print("Loading preprocessed data...")
# (Same data loading logic as before)
X_train = pd.read_csv('data/processed/X_train_processed.csv')
X_test = pd.read_csv('data/processed/X_test_processed.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# --- 4. Define, Compile, and Summarize Model ---
n_features = X_train.shape[1]
model = keras.Sequential([
    layers.Input(shape=(n_features,)),
    layers.Dense(64, activation='relu', name='Hidden_Layer_1'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu', name='Hidden_Layer_2'),
    layers.Dense(1, activation='sigmoid', name='Output_Layer')
], name="MLP_Model")

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
model.summary()

# --- 5. Visualize and Save Model Architecture ---
if CAN_PLOT_MODEL:
    plot_model(model, to_file=f'{RESULTS_DIR}/model_architecture.png', show_shapes=True, show_layer_names=True)
    print(f"Model architecture diagram saved to '{RESULTS_DIR}/model_architecture.png'")

# --- 6. Train the Model ---
print("\nTraining the MLP model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=256, validation_split=0.2, verbose=1)

# --- 7. Plot and Save Training History ---
pd.DataFrame(history.history).plot(figsize=(10, 7))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.title("MLP Model Training History")
plt.savefig(f'{RESULTS_DIR}/training_history.png')
print(f"Training history plot saved to '{RESULTS_DIR}/training_history.png'")

# --- 8. Evaluate and Save Report ---
print("\nEvaluating the model on the test set...")
y_pred_proba = model.predict(X_test).ravel()
y_pred = (y_pred_proba > 0.5).astype(int)
report = classification_report(y_test, y_pred, target_names=['No (deposit)', 'Yes (deposit)'])
print("\nClassification Report:")
print(report)
with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(report)
print(f"\nClassification report saved to '{RESULTS_DIR}/classification_report.txt'")

# --- 9. Visualize and Save Evaluation Plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax1, cmap='Reds')
ax1.set_title("Confusion Matrix - MLP")
RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax2)
ax2.set_title("ROC Curve - MLP")
plt.savefig(f'{RESULTS_DIR}/evaluation_plots.png')
print(f"Evaluation plots saved to '{RESULTS_DIR}/evaluation_plots.png'")
plt.close('all')

# --- 10. Save the Trained Model ---
print("\nSaving the trained model...")
model.save(f'{MODELS_DIR}/model.keras')
print(f"Model saved to '{MODELS_DIR}/model.keras'")

print("\nDeep Learning modeling complete!")