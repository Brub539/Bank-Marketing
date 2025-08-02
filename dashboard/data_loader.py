import streamlit as st
import pandas as pd
import joblib
import glob
import os
from tensorflow.keras.models import load_model

@st.cache_data(show_spinner="Loading data and models...")
def load_all():
    """
    Loads all necessary data and trained models.
    The @st.cache_data decorator ensures this function only runs once.
    """
    # --- Load Data ---
    X_test = pd.read_csv('data/processed_target_encoding/X_test_processed.csv')
    y_test = pd.read_csv('data/processed_target_encoding/y_test.csv').values.ravel()
    raw_df = pd.read_csv('data/raw/bank-additional-full.csv', sep=';')

    # --- Load Models ---
    models = {}
    FOLDERS_TO_IGNORE = ['best_params', 'preprocessor_target_encoding']
    model_folders = sorted([f for f in os.listdir('models') if os.path.isdir(os.path.join('models', f)) and f not in FOLDERS_TO_IGNORE])
    
    for folder in model_folders:
        try:
            model_path_search = glob.glob(f'models/{folder}/model.*')
            if not model_path_search: continue
            
            model_path = model_path_search[0]
            model_name = folder.replace('_', ' ').title()
            
            if model_path.endswith('.keras'):
                models[model_name] = load_model(model_path)
            else:
                models[model_name] = joblib.load(model_path)
        except Exception as e:
            print(f"Skipping model {folder} due to loading error: {e}")
            
    return X_test, y_test, raw_df, models