import os
import subprocess
import re
import numpy as np

def run_experiment():
    seeds = [42, 1337, 2024, 101, 999]
    accuracies = []
    aucs = []

    for seed in seeds:
        print(f"=== Running experiment with seed {seed} ===")
        env = os.environ.copy()
        env["RUN_SEED"] = str(seed)
        
        print("  1/3 Assembling data...")
        subprocess.run(["python3", "src/task0_assemble.py"], env=env, check=True, stdout=subprocess.DEVNULL)
        
        print("  2/3 Extracting features...")
        subprocess.run(["python3", "-m", "src.task1_stylometry"], env=env, check=True, stdout=subprocess.DEVNULL)
        
        print("  3/3 Training Tier C...")
        result = subprocess.run(["python3", "-m", "src.task2_tier_c"], env=env, check=True, capture_output=True, text=True)
        
        # Parse log for metrics
        log = result.stdout
        acc_match = re.search(r"accuracy.*?([0-9.]+)\s+\d+\n\s+macro", log)
        auc_match = re.search(r"Test macro AUC = ([0-9.]+)", log)
        
        acc = float(acc_match.group(1)) if acc_match else None
        auc = float(auc_match.group(1)) if auc_match else None
        
        print(f"  -> Accuracy: {acc}, Macro AUC: {auc}")
        
        if acc is not None: accuracies.append(acc)
        if auc is not None: aucs.append(auc)

    print("\n=== FINAL RESULTS (Tier C across 5 seeds) ===")
    print(f"Mean Accuracy : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Mean Macro AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

if __name__ == "__main__":
    run_experiment()
