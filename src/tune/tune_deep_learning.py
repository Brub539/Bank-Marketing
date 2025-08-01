# --- 1. Import Libraries ---
import pandas as pd
import optuna
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import warnings
import sys
import joblib

warnings.filterwarnings('ignore')

# --- 2. Load Processed Data ---
print("\n" + "="*70)
print("--- Starting Deep Learning Hyperparameter Tuning ---")
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
    print(f"Error: Data not found. Please run the pipeline first.")
    sys.exit()

# --- 3. Apply SMOTE (once, outside the tuning loop for speed) ---
print("Applying SMOTE to the training data...")
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print(f"Resampled training shape: {X_train_res.shape}")

# --- 4. Define the Model Building Function for Optuna ---
def create_model(trial, n_features):
    """Builds a Keras model with hyperparameters suggested by Optuna."""
    # Tune the number of hidden layers
    n_layers = trial.suggest_int('n_layers', 1, 3)
    # Tune the initial learning rate
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(n_features,)))
    
    # Create the hidden layers
    for i in range(n_layers):
        # Tune the number of neurons and dropout rate for each layer
        n_units = trial.suggest_int(f'n_units_l{i}', 32, 256, log=True)
        dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.2, 0.5)
        
        model.add(layers.Dense(n_units))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.Dropout(dropout_rate))
        
    model.add(layers.Dense(1, activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])
    return model

# --- 5. Define the Objective Function for Optuna ---
def objective(trial):
    """The function for Optuna to optimize."""
    n_features = X_train_res.shape[1]
    model = create_model(trial, n_features)
    
    # Use early stopping to prevent wasting time on bad trials
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    
    model.fit(
        X_train_res, y_train_res,
        epochs=50, # Train for a moderate number of epochs for speed
        batch_size=256,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=0 # Suppress epoch-by-epoch logging for cleaner tuning output
    )
    
    # Evaluate using F1-score, which is what we care about
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    return f1

# --- 6. Run the Optimization ---
print("\nStarting optimization with Optuna... (This will take a significant amount of time)")
study = optuna.create_study(direction='maximize')
# Let's run for 20 trials as a deep learning trial is slow
study.optimize(objective, n_trials=20)

# --- 7. Display and Save the Results ---
print("\nOptimization finished.")
print("\n--- Best Trial ---")
trial = study.best_trial
print(f"  Value (F1-score): {trial.value:.4f}")
print("  Params (Architecture Blueprint): ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

best_params_file = 'models/best_params/best_dl_params.joblib'
os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
joblib.dump(trial.params, best_params_file)
print(f"\nBest DL architecture blueprint saved to '{best_params_file}'")
print("\n--- Deep Learning Tuning Complete ---")