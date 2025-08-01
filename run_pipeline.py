# --- 1. Import Libraries ---
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import os
import joblib
import numpy as np
from category_encoders import TargetEncoder

# --- 2. Setup Project Paths ---
os.makedirs('data/processed_target_encoding', exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- 3. Data Loading ---
def load_data(file_path='data/raw/bank-additional-full.csv'):
    """Loads the new, richer bank marketing dataset."""
    if not os.path.exists(file_path):
        # Allow master script to handle errors, but notify here
        print(f"CRITICAL ERROR in run_pipeline: Dataset not found at {file_path}")
        return None
    df = pd.read_csv(file_path, sep=';')
    return df

# --- 4. Main Preprocessing Logic ---
def preprocess_data(df):
    """Preprocesses the new dataset with all feature engineering steps."""
    # --- Outlier Capping ---
    numeric_cols_to_cap = ['age', 'duration', 'campaign', 'cons.conf.idx']
    for col in numeric_cols_to_cap:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # --- Group Aggregate Features ---
    numeric_to_agg = {
        'age': ['mean', 'std'], 
        'cons.price.idx': ['mean', 'std'], 
        'euribor3m': ['mean', 'std']
    }
    groups = ['job', 'education', 'marital']
    for group_col in groups:
        for num_col, aggs in numeric_to_agg.items():
            for agg_type in aggs:
                new_col_name = f'{agg_type}_{num_col}_by_{group_col}'
                df[new_col_name] = df.groupby(group_col)[num_col].transform(agg_type)
    df.fillna(0, inplace=True)

    # --- Feature Creation and Data Splitting ---
    df['was_contacted_before'] = (df['pdays'] != 999).astype(int)
    X = df.drop('y', axis=1)
    y = df['y'].map({'yes': 1, 'no': 0})
    X = X.drop(['duration', 'pdays'], axis=1)

    # --- Time-Based Split ---
    split_index = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # --- Preprocessing Pipeline ---
    numeric_features = X_train.select_dtypes(include=np.number).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', TargetEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )
    preprocessor.fit(X_train, y_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    joblib.dump(preprocessor, 'models/preprocessor_target_encoding.joblib')

    # --- Final DataFrame Creation ---
    processed_cols = preprocessor.get_feature_names_out()
    X_train_df = pd.DataFrame(X_train_processed, columns=processed_cols, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=processed_cols, index=X_test.index)

    return X_train_df, X_test_df, y_train, y_test

# --- 5. Execution ---
if __name__ == "__main__":
    raw_df = load_data()
    if raw_df is not None:
        X_train_df, X_test_df, y_train_data, y_test_data = preprocess_data(raw_df)
        
        output_dir = 'data/processed_target_encoding'
        X_train_df.to_csv(f'{output_dir}/X_train_processed.csv', index=False)
        X_test_df.to_csv(f'{output_dir}/X_test_processed.csv', index=False)
        y_train_data.to_csv(f'{output_dir}/y_train.csv', index=False)
        y_test_data.to_csv(f'{output_dir}/y_test.csv', index=False)
        
        print(f"Pipeline complete. Processed data saved to '{output_dir}/'.")