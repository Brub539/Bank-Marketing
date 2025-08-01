# --- 1. Import Libraries ---
import pandas as pd
import lightgbm as lgb
import optuna
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score
import warnings
import sys
import joblib
from datetime import datetime

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- 2. Load Processed Data ---
print("\n--- Starting Hyperparameter Tuning ---")
print("Loading preprocessed data (TARGET ENCODED)...")
# Note: The path is now relative to the root directory, not the notebooks folder
data_dir = 'data/processed_target_encoding'
try:
    X_train = pd.read_csv(f'{data_dir}/X_train_processed.csv')
    y_train = pd.read_csv(f'{data_dir}/y_train.csv').values.ravel()
    print(f"Data loaded successfully. Training data shape: {X_train.shape}")
except FileNotFoundError:
    print("Error: Target encoded data not found.")
    print("Please ensure you have run the correct pipeline script first.")
    sys.exit()

# --- 3. Define the Hyperparameter Tuning Objective Function ---
def objective(trial, X, y):
    """
    Defines the objective for Optuna to optimize.
    A 'trial' is a single run with a specific set of hyperparameters.
    """
    params = {
        'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
        'boosting_type': 'gbdt', 'n_estimators': 1000, 'class_weight': 'balanced',
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    N_SPLITS = 5
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    f1_scores = []

    for train_index, val_index in tscv.split(X, y):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]

        model = lgb.LGBMClassifier(**params)
        early_stopping_callback = lgb.early_stopping(100, verbose=False)
        
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],
                  eval_metric='f1', callbacks=[early_stopping_callback])

        preds = model.predict(X_val_fold)
        f1 = f1_score(y_val_fold, preds, pos_label=1)
        f1_scores.append(f1)

    return sum(f1_scores) / len(f1_scores)

# --- 4. Run the Optimization ---
print("\nStarting hyperparameter tuning with TimeSeriesSplit Cross-Validation...")
# Optuna will now show its progress bar directly in the terminal
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=50)

# --- 5. Display and Save the Results ---
print("\nOptimization finished.")
print("Number of finished trials: ", len(study.trials))
print("\n--- Best Trial ---")
trial = study.best_trial
print(f"  Value (F1-score): {trial.value:.4f}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_params_file = 'models/best_params/best_lgbm_params.joblib'
os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
joblib.dump(trial.params, best_params_file)
print(f"\nBest parameters saved to '{best_params_file}'")
print("\n--- Hyperparameter Tuning Complete ---")