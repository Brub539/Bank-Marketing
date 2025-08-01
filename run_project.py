import os
import subprocess
import pandas as pd
import glob
import warnings

warnings.filterwarnings('ignore')

# --- 1. Define the Project Workflow in STAGES ---
# --- KEY CHANGE: The output file now points to the SAVED MODEL file ---
# This is the true final artifact of each script.

# STAGE 1: Data Preparation
PIPELINE_WORK = {
    'run_pipeline.py': 'data/processed_target_encoding/X_train_processed.csv'
}

# STAGE 2: Hyperparameter Tuning
TUNING_WORK = {
    'src/tune/tune_LGBM.py': 'models/best_params/best_lgbm_params.joblib',
    'src/tune/tune_random_forest.py': 'models/best_params/best_rf_params.joblib',
    'src/tune/tune_deep_learning.py': 'models/best_params/best_dl_params.joblib'
}

# STAGE 3: Final Model Training
TRAINING_WORK = {
    'src/train_baseline.py': 'models/logistic_regression_smote_te/model.joblib',
    'src/train_random_forest.py': 'models/rf_tuned_class_weight/model.joblib',
    'src/train_deep_learning.py': 'models/deep_learning_mlp_tuned/model.keras',
    'src/train_catboost.py': 'models/catboost_target_encoding/model.joblib',
    'src/train_tabnet.py': 'models/tabnet_smote_target_encoding/model.joblib',
    'src/train_voting_model.py': 'models/voting_ensemble_smote_v2/model.joblib',
    'src/train_stacking_model.py': 'models/stacking_ensemble_smote_v2/model.joblib',
    'src/train_lightgbm.py': 'models/lightgbm_rfe_smote_target_encoding/model.joblib'
}
# --- END OF KEY CHANGE ---


# --- 2. Upgraded Helper Function with Caching (remains the same) ---
def execute_command(script_path, output_file, description):
    """Runs a script, but only if its target output file does not already exist."""
    print("=" * 70)
    print(f"Checking task: {description}")
    
    if os.path.exists(output_file):
        print(f"Output '{output_file}' already exists. SKIPPING task.")
        return

    print(f"Executing: {description}")
    command = ['python', script_path]
    print(f"Command: {' '.join(command)}")
    print("-" * 70)
    try:
        subprocess.run(command, check=True, text=True, encoding='utf-8')
        print(f"\n--- Successfully executed {description} ---\n")
    except Exception as e:
        print(f"--- Error executing {description} ---")
        print("See the output above for the error details.")
        raise

# --- 3. Analysis Function (remains the same) ---
def analyze_results():
    # (This function is perfect as-is)
    print("=" * 70)
    print("FINAL ANALYSIS: Gathering and Comparing All Model Results")
    print("=" * 70)
    report_files = glob.glob('results/*/classification_report.txt')
    if not report_files:
        print("No classification reports found.")
        return
    results = []
    for report_file in report_files:
        try:
            model_name = os.path.basename(os.path.dirname(report_file))
            with open(report_file, 'r') as f:
                for line in f:
                    if 'Yes (deposit)' in line:
                        parts = line.split()
                        precision, recall, f1_score = float(parts[2]), float(parts[3]), float(parts[4])
                        results.append({'Model': model_name.replace('_', ' ').title(), 'Precision (Yes)': precision, 'Recall (Yes)': recall, 'F1-Score (Yes)': f1_score})
                        break
        except Exception as e:
            print(f"Could not parse report '{report_file}': {e}")
    if not results:
        print("Could not extract results.")
        return
    comparison_df = pd.DataFrame(results).sort_values(by='F1-Score (Yes)', ascending=False).reset_index(drop=True)
    print("\n--- Final Model Performance Comparison ---")
    print(comparison_df.to_string())
    winner = comparison_df.iloc[0]
    print("\n--- Conclusion ---")
    print(f"The champion model is '{winner['Model']}' with a final F1-Score of {winner['F1-Score (Yes)']:.4f}.")


# --- 4. Main Orchestrator (remains the same) ---
if __name__ == "__main__":
    try:
        # STAGE 1: Data Preparation
        print("\n" + "#"*70)
        print("### STAGE 1: DATA PREPARATION ###")
        print("#"*70)
        for script, output in PIPELINE_WORK.items():
            execute_command(script, output, "Task: Data Preprocessing")
            
        # STAGE 2: Hyperparameter Tuning
        print("\n" + "#"*70)
        print("### STAGE 2: HYPERPARAMETER TUNING ###")
        print("#"*70)
        for script, output in TUNING_WORK.items():
            description = script.split('/')[-1].replace('.py', '').replace('_', ' ').title()
            execute_command(script, output, f"Task: {description}")

        # STAGE 3: Final Model Training
        print("\n" + "#"*70)
        print("### STAGE 3: FINAL MODEL TRAINING ###")
        print("#"*70)
        for script, output in TRAINING_WORK.items():
            description = script.split('/')[-1].replace('.py', '').replace('_', ' ').title()
            execute_command(script, output, f"Task: {description}")
            
        # STAGE 4: The final analysis always runs
        print("\n" + "#"*70)
        print("### STAGE 4: FINAL ANALYSIS ###")
        print("#"*70)
        analyze_results()

    except Exception as e:
        print(f"\n\nAn error occurred during the project execution.")
        print("The process was halted.")