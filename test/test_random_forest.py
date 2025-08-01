import subprocess
import os

def execute_command(command, description):
    print("=" * 70)
    print(f"Executing: {description}")
    print(f"Command: {' '.join(command)}")
    print("=" * 70)
    try:
        subprocess.run(command, check=True, text=True, encoding='utf-8')
        print(f"\n--- Successfully executed {description} ---\n")
        return True
    except Exception as e:
        print(f"--- Error executing {description} ---")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("--- Starting Targeted Test for Random Forest ---\n")
    
    rf_tuner_script = 'src/tune_random_forest.py'
    rf_trainer_script = 'src/train_random_forest.py'
    
    # STEP 1: Run the tuner to create the needed .joblib file
    success = execute_command(
        ['python', rf_tuner_script], 
        f"Prerequisite: Running RF Tuner ({os.path.basename(rf_tuner_script)})"
    )
    
    # STEP 2: If the tuner was successful, run the main trainer script
    if success:
        execute_command(
            ['python', rf_trainer_script], 
            f"Main Test: Running RF Trainer ({os.path.basename(rf_trainer_script)})"
        )
    else:
        print("\nHalting test because the prerequisite tuning script failed.")

    print("\n--- Targeted Test for Random Forest Complete ---")