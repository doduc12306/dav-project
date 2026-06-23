import subprocess
import sys
import os

def run_script(cmd):
    print(f"\n==========================================")
    print(f"Running: {' '.join(cmd)}")
    print(f"==========================================\n")
    try:
        process = subprocess.run(cmd, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def main():
    print("=== S-JEPA SKELETON ACTION RECOGNITION DEMO PIPELINE ===")
    
    # Step 1: Run Exploratory Data Analysis (EDA) & Mock Generator
    print("\n--- [STEP 1] Running EDA & Generating Mock Dataset ---")
    if not run_script([sys.executable, "src/eda.py"]):
        print("EDA failed. Aborting.")
        return
        
    # Step 2: Run 3D Skeleton Plotly Visualizer
    print("\n--- [STEP 2] Running 3D Skeleton Plotly Visualizer ---")
    if not run_script([sys.executable, "src/visualize.py"]):
        print("3D Visualization failed. Aborting.")
        return
        
    # Step 3: Run S-JEPA Self-Supervised Pre-training (Quick test: 3 epochs)
    print("\n--- [STEP 3] Running S-JEPA Pre-training (3 Epochs) ---")
    if not run_script([sys.executable, "src/train.py", "--epochs", "3", "--batch_size", "4", "--limit", "150"]):
        print("S-JEPA Pre-training failed. Aborting.")
        return
        
    # Step 4: Run Downstream Linear Probing & Evaluation (Quick test: 2 epochs)
    print("\n--- [STEP 4] Running Downstream Evaluation (t-SNE & Confusion Matrix) ---")
    if not run_script([sys.executable, "src/downstream.py", "--epochs", "2", "--batch_size", "4", "--limit", "150"]):
        print("Downstream Evaluation failed. Aborting.")
        return

    print("\n==========================================")
    print("DEMO PIPELINE COMPLETED SUCCESSFULLY!")
    print("All generated plots can be found in the './plots/' directory.")
    print("Model checkpoints are saved in the './checkpoints/' directory.")
    print("==========================================")

if __name__ == "__main__":
    main()
