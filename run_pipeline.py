# --- 1. Import Libraries ---
import pandas as pd
from zipfile import ZipFile
from urllib.request import urlopen
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import joblib

# --- 2. Setup Project Paths ---
os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- 3. Data Loading ---
def load_data(zip_url, zip_path='data/bank.zip'):
    """Downloads and loads the bank marketing dataset."""
    if not os.path.exists(zip_path):
        print(f"Downloading data from {zip_url}...")
        with urlopen(zip_url) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        print("Download complete.")

    with ZipFile(zip_path, 'r') as zip_ref:
        with zip_ref.open('bank-full.csv') as csv_file:
            df = pd.read_csv(csv_file, sep=';')
    print("Dataset loaded successfully!")
    return df

# --- 4. Main Preprocessing Logic ---
def preprocess_data(df):
    """Preprocesses the raw bank marketing data with feature engineering."""
    print("Starting data preprocessing...")

    # --- NEW: ADVANCED FEATURE ENGINEERING ---
    # Create a binary feature indicating if the client was contacted before.
    # The value 999 in 'pdays' means the client was not previously contacted.
    df['was_contacted_before'] = (df['pdays'] != 999).astype(int)
    print("New feature 'was_contacted_before' created from 'pdays'.")

    # Separate Features (X) and Target (y)
    X = df.drop('y', axis=1)
    y = df['y'].map({'yes': 1, 'no': 0})

    # Drop 'duration' to prevent data leakage and 'pdays' as it's now encoded.
    X = X.drop(['duration', 'pdays'], axis=1)
    print("Dropped 'duration' and 'pdays' columns.")

    # Identify feature types (the new feature is already numeric)
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'int32']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # --- UPDATED: TRAIN-TEST SPLIT ---
    # Using a 1/3 test size as requested.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1/3, random_state=42, stratify=y
    )
    print("Data split into 2/3 training and 1/3 testing sets.")

    # Create the preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Fit the preprocessor on the training data and transform both sets
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the fitted preprocessor
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("Preprocessor saved to 'models/preprocessor.joblib'")

    print("Preprocessing complete.")
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

# --- 5. Execution ---
if __name__ == "__main__":
    ZIP_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip'
    
    raw_df = load_data(ZIP_URL, 'data/bank.zip')
    X_train_proc, X_test_proc, y_train_data, y_test_data, proc_pipeline = preprocess_data(raw_df)
    
    # Get feature names after transformation for verification
    new_feature_names = proc_pipeline.get_feature_names_out()

    print(f"\nProcessed Training Data Shape: {X_train_proc.shape}")
    print(f"Processed Test Data Shape: {X_test_proc.shape}")
    print(f"Number of features after preprocessing: {len(new_feature_names)}")
    
    # Save processed data
    # Note: We save as DataFrames to preserve column names if needed, though they are generic here.
    pd.DataFrame(X_train_proc, columns=new_feature_names).to_csv('data/processed/X_train_processed.csv', index=False)
    pd.DataFrame(X_test_proc, columns=new_feature_names).to_csv('data/processed/X_test_processed.csv', index=False)
    y_train_data.to_csv('data/processed/y_train.csv', index=False)
    y_test_data.to_csv('data/processed/y_test.csv', index=False)
    print("\nNew processed data saved to 'data/processed/' directory.")