import os
import subprocess
import pandas as pd
import glob
import warnings

# Suppress warnings from libraries for cleaner output
warnings.filterwarnings('ignore')

# --- 1. Define the Project Workflow ---
PIPELINE_SCRIPT = 'run_pipeline.py'
# --- KEY CHANGE: We now call a .py script for tuning ---
TUNING_SCRIPT = 'src/tune_hyperparameters.py'
# --- END OF CHANGE ---

TRAINING_SCRIPTS = [
    'src/train_baseline.py',
    'src/train_random_forest.py',
    'src/train_deep_learning.py',
    'src/train_catboost.py',
    'src/train_tabnet.py',
    'src/train_voting_model.py',
    'src/train_stacking_model.py',
    'src/train_lightgbm.py'
]

# --- 2. Helper Function to Run Scripts (remains the same) ---
def execute_command(command, description):
    """A helper function to run a command and print its status."""
    print("=" * 70)
    print(f"Executing: {description}")
    print(f"Command: {' '.join(command)}")
    print("=" * 70)
    try:
        process = subprocess.run(
            command, 
            check=True, 
            text=True, 
            encoding='utf-8'
        )
        print(f"\n--- Successfully executed {description} ---\n")
    except FileNotFoundError:
        print(f"Error: Command not found. Is '{command[0]}' installed and in your PATH?")
        raise
    except subprocess.CalledProcessError as e:
        print(f"--- Error executing {description} ---")
        print("See the output above for the error details.")
        raise

# --- 3. Function to Gather and Analyze Results (remains the same) ---
def analyze_results():
    """Finds all classification reports, parses them, and prints a comparison table."""
    print("=" * 70)
    print("FINAL ANALYSIS: Gathering and Comparing All Model Results")
    print("=" * 70)
    
    report_files = glob.glob('results/*/classification_report.txt')
    
    if not report_files:
        print("No classification reports found. Cannot perform analysis.")
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
                        results.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Precision (Yes)': precision,
                            'Recall (Yes)': recall,
                            'F1-Score (Yes)': f1_score
                        })
                        break
        except Exception as e:
            print(f"Could not parse report '{report_file}': {e}")

    if not results:
        print("Could not extract results from any report files.")
        return

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(by='F1-Score (Yes)', ascending=False).reset_index(drop=True)
    
    print("\n--- Final Model Performance Comparison ---")
    print(comparison_df.to_string())

    winner = comparison_df.iloc[0]
    print("\n--- Conclusion ---")
    print(f"The champion model is '{winner['Model']}' with a final F1-Score of {winner['F1-Score (Yes)']:.4f}.")
    print("This demonstrates the power of the chosen feature engineering, imbalance handling, and modeling techniques.")
    print("The project is complete and all artifacts (models, results, processed data) are saved.")

# --- 4. Main Orchestrator ---
if __name__ == "__main__":
    try:
        # STEP 1: Run the data engineering pipeline
        execute_command(['python', PIPELINE_SCRIPT], "Data Preprocessing Pipeline")
        
        # --- KEY CHANGE: Execute the tuning SCRIPT, not the notebook ---
        # STEP 2: Run the new hyperparameter tuning script
        execute_command(['python', TUNING_SCRIPT], "Hyperparameter Tuning")
        # --- END OF CHANGE ---

        # STEP 3: Run all training scripts
        for script in TRAINING_SCRIPTS:
            execute_command(['python', script], f"Training Script: {os.path.basename(script)}")
        
        # STEP 4: Run the final analysis
        analyze_results()
    except Exception as e:
        print(f"\n\nAn error occurred during the project execution.")
        print("The process was halted.")