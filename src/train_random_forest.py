# --- 1. Import Libraries ---
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import os

# --- 2. Setup Paths and Directories ---
MODEL_NAME = 'random_forest'
RESULTS_DIR = f'results/{MODEL_NAME}'
MODELS_DIR = f'models/{MODEL_NAME}'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 3. Load Processed Data ---
print("Loading preprocessed data...")
# (Same data loading logic as before)
X_train = pd.read_csv('data/processed/X_train_processed.csv')
X_test = pd.read_csv('data/processed/X_test_processed.csv')
y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

# --- 4. Train Random Forest Model ---
print("Training Random Forest model...")
rf_clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)

# --- 5. Evaluate the Model ---
print("Evaluating the model...")
y_pred = rf_clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['No (deposit)', 'Yes (deposit)'])
print("\nClassification Report:")
print(report)

# Save the report to the model's specific results directory
with open(f'{RESULTS_DIR}/classification_report.txt', 'w') as f:
    f.write(report)
print(f"\nClassification report saved to '{RESULTS_DIR}/classification_report.txt'")

# --- 6. Visualize and Save Plots ---
# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(rf_clf, X_test, y_test, ax=ax, cmap='Greens')
ax.set_title("Confusion Matrix - Random Forest")
plt.savefig(f'{RESULTS_DIR}/confusion_matrix.png')
print(f"Confusion matrix plot saved to '{RESULTS_DIR}/confusion_matrix.png'")

# Plot ROC Curve
fig, ax = plt.subplots(figsize=(8, 6))
RocCurveDisplay.from_estimator(rf_clf, X_test, y_test, ax=ax)
ax.set_title("ROC Curve - Random Forest")
plt.savefig(f'{RESULTS_DIR}/roc_curve.png')
print(f"ROC curve plot saved to '{RESULTS_DIR}/roc_curve.png'")
plt.close('all')

# --- 7. Save the Trained Model ---
print("Saving the trained model...")
joblib.dump(rf_clf, f'{MODELS_DIR}/model.joblib')
print(f"Model saved to '{MODELS_DIR}/model.joblib'")

print("\nRandom Forest modeling complete!")