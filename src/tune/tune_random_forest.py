# --- 1. Import Libraries ---
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import optuna
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import warnings
import sys
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- 2. Load Processed Data ---
print("\n" + "="*70)
print("--- Starting Random Forest Hyperparameter Tuning ---")
print("="*70)
print("Loading preprocessed data (TARGET ENCODED)...")
data_dir = 'data/processed_target_encoding'
try:
    X_train = pd.read_csv(f'{data_dir}/X_train_processed.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    print(f"Data loaded successfully. Training data shape: {X_train.shape}")
except FileNotFoundError:
    print(f"Error: Target encoded data not found in '{data_dir}'.")
    print("Please run the pipeline script first.")
    sys.exit()

# --- 3. Define the Hyperparameter Tuning Objective Function ---
def objective(trial, X, y):
    """Defines the objective for Optuna to optimize for RandomForestClassifier."""
    # Define a search space for key RandomForest hyperparameters
    # These parameters help prevent the model from getting too complex and overfitting.
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
        'max_depth': trial.suggest_int('max_depth', 5, 40),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 32),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
        'max_features': trial.suggest_float('max_features', 0.2, 0.8),
        'class_weight': 'balanced_subsample', # Excellent choice for RF
        'n_jobs': -1,
        'random_state': 42
    }

    # Using TimeSeriesSplit for robust validation on our time-ordered data
    N_SPLITS = 5
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    f1_scores = []
    
    model = RandomForestClassifier(**params)
    
    # We will fit on expanding windows of time-series data
    for train_index, val_index in tscv.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        model.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_val_fold)
        f1 = f1_score(y_val_fold, preds, pos_label=1)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)

# --- 4. Run the Optimization ---
print("\nStarting optimization with Optuna...")
study = optuna.create_study(direction='maximize')
# We'll run fewer trials than for LightGBM as RF can be slower
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)

# --- 5. Display and Save the Results ---
print("\nOptimization finished.")
print("\n--- Best Trial ---")
trial = study.best_trial
print(f"  Value (F1-score): {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# This is the file that train_random_forest.py is looking for
best_params_file = 'models/best_params/best_rf_params.joblib'
os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
joblib.dump(trial.params, best_params_file)
print(f"\nBest Random Forest parameters saved to '{best_params_file}'")

print("\n" + "="*70)
print("--- Random Forest Tuning Complete ---")
print("="*70)