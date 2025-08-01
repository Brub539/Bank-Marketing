import subprocess
import os
import sys

# ==============================================================================
# TARGETED TEST SCRIPT FOR THE DEEP LEARNING PIPELINE
# ==============================================================================
# This script is designed to test the full deep learning workflow:
# 1. Hyperparameter tuning for the model architecture.
# 2. Training the final model using the tuned architecture.
#
# It assumes 'run_pipeline.py' has already been run successfully at least once.
# ==============================================================================

def execute_command(command, description):
    """A helper function to run a command and print its status."""
    print("=" * 70)
    print(f"Executing: {description}")
    print(f"Command: {' '.join(command)}")
    print("=" * 70)
    try:
        # We need to run python from the parent directory for paths to work correctly
        # The `..` in the python -m command achieves this.
        subprocess.run(command, check=True, text=True, encoding='utf-8')
        print(f"\n--- Successfully executed {description} ---\n")
        return True
    except Exception as e:
        print(f"--- Error executing {description} ---")
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("--- Starting Targeted Test for Deep Learning Models ---\n")
    
    # Define the two scripts we need to run in order
    dl_tuner_script = 'src/tune/tune_deep_learning.py'
    dl_trainer_script = 'src/train_deep_learning.py'
    
    # Verify that the necessary scripts exist before starting
    if not os.path.exists(dl_tuner_script) or not os.path.exists(dl_trainer_script):
        print(f"Error: Could not find one of the required scripts.")
        print(f"  - Check for: {dl_tuner_script}")
        print(f"  - Check for: {dl_trainer_script}")
        sys.exit()

    # STEP 1: Run the Deep Learning hyperparameter tuner.
    success = execute_command(
        ['python', dl_tuner_script], 
        f"Prerequisite: Running DL Tuner ({dl_tuner_script})"
    )
    
    # STEP 2: If the tuner was successful, run the main training script.
    if success:
        execute_command(
            ['python', dl_trainer_script], 
            f"Main Test: Running DL Trainer ({dl_trainer_script})"
        )
    else:
        print("\nHalting test because the prerequisite tuning script failed.")

    print("\n--- Targeted Test for Deep Learning Complete ---")