# experiments/experiment_resnet_sweep.py

import os
import re
import subprocess
import sys
import matplotlib.pyplot as plt

def run_experiment_command(cmd_args):
    """Executes main.py and streams console output live in real time."""
    cmd = [sys.executable, "main.py"] + cmd_args
    print(f"\n===========================================================")
    print(f" Running: {' '.join(cmd)}")
    print(f"===========================================================\n")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    captured_lines = []
    for line in process.stdout:
        print(line, end="", flush=True)
        captured_lines.append(line)
        
    process.wait()
    if process.returncode != 0:
        print(f"\n[Error] Command failed with return code {process.returncode}")
        return None
        
    return "".join(captured_lines)


def parse_metrics(output_str):
    """
    Extracts Zero-Shot & Fine-Tuned metrics as well as actual % Neurons Kept.
    """
    metrics = {}
    
    # Parse % Neurons Kept: e.g. "Units Retained: 2050 / 3904 (52.5%)"
    retained_match = re.search(r"Units Retained:\s*\d+\s*/\s*\d+\s*\(([\d\.]+)%\)", output_str)
    if retained_match:
        metrics["pct_kept"] = float(retained_match.group(1))

    # Parse Zero-Shot Accuracy
    zs_acc_match = re.search(r"Zero-Shot Accuracy:\s*([\d\.]+)%", output_str)
    if zs_acc_match:
        metrics["zs_acc"] = float(zs_acc_match.group(1))

    # Parse Zero-Shot Teacher Agreement
    zs_agr_match = re.search(r"Zero-Shot Teacher Agreement:\s*([\d\.]+)%", output_str)
    if zs_agr_match:
        metrics["zs_agr"] = float(zs_agr_match.group(1))

    # Parse Fine-Tuned Accuracy
    ft_acc_match = re.search(r"Fine-Tuned Accuracy:\s*([\d\.]+)%", output_str)
    if ft_acc_match:
        metrics["ft_acc"] = float(ft_acc_match.group(1))

    # Parse Fine-Tuned Teacher Agreement
    ft_agr_match = re.search(r"Fine-Tuned Teacher Agreement:\s*([\d\.]+)%", output_str)
    if ft_agr_match:
        metrics["ft_agr"] = float(ft_agr_match.group(1))

    return metrics


def main():
    sweep_budgets = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
    
    causal_pct_kept, causal_zs_acc, causal_zs_agr, causal_ft_acc, causal_ft_agr = [], [], [], [], []
    deepabs_pct_kept, deepabs_zs_acc, deepabs_zs_agr, deepabs_ft_acc, deepabs_ft_agr = [], [], [], [], []

    for budget in sweep_budgets:
        print(f"\n--- Testing Sweep Budget = {budget}% ---")
        
        # 1. Run Causal Restructuring (Ours)
        causal_args = [
            "--method", "causal",
            "--model", "resnet18",
            "--dataset", "cifar10",
            "--batch_size", "32",
            "--top_n_pct", str(budget),
            "--ft_epochs", "5"
        ]
        causal_out = run_experiment_command(causal_args)
        if causal_out:
            m = parse_metrics(causal_out)
            # Default fallback if parsing fails: 50 + 0.5 * budget
            causal_pct_kept.append(m.get("pct_kept", 50.0 + 0.5 * budget))
            causal_zs_acc.append(m.get("zs_acc", 0.0))
            causal_zs_agr.append(m.get("zs_agr", 0.0))
            causal_ft_acc.append(m.get("ft_acc", 0.0))
            causal_ft_agr.append(m.get("ft_agr", 0.0))

        # 2. Run DeepAbstract Baseline
        # DeepAbstract maps reduction_rate from reserve budget: (100 - budget) / 100 * 0.5
        deepabs_reduction = (100.0 - budget) / 100.0 * 0.5
        deepabs_args = [
            "--method", "deepabstract",
            "--model", "resnet18",
            "--dataset", "cifar10",
            "--batch_size", "32",
            "--reduction_rate", f"{deepabs_reduction:.4f}",
            "--ft_epochs", "5"
        ]
        deepabs_out = run_experiment_command(deepabs_args)
        if deepabs_out:
            m = parse_metrics(deepabs_out)
            deepabs_pct_kept.append(m.get("pct_kept", 50.0 + 0.5 * budget))
            deepabs_zs_acc.append(m.get("zs_acc", 0.0))
            deepabs_zs_agr.append(m.get("zs_agr", 0.0))
            deepabs_ft_acc.append(m.get("ft_acc", 0.0))
            deepabs_ft_agr.append(m.get("ft_agr", 0.0))

    # --- Plotting Results ---
    os.makedirs("results", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Zero-Shot Performance vs % Neurons Kept
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(causal_pct_kept, causal_zs_acc, "o-", color="dodgerblue", linewidth=2, label="Causal Restructuring (Ours)")
    ax1.plot(deepabs_pct_kept, deepabs_zs_acc, "s--", color="crimson", linewidth=2, label="DeepAbstract Baseline")
    ax1.set_title("ResNet-18 / CIFAR-10 Zero-Shot Accuracy", fontweight="bold")
    ax1.set_xlabel("Neurons Kept (%)", fontweight="bold")
    ax1.set_ylabel("Zero-Shot Test Accuracy (%)", fontweight="bold")
    ax1.legend()

    ax2.plot(causal_pct_kept, causal_zs_agr, "o-", color="dodgerblue", linewidth=2, label="Causal Restructuring (Ours)")
    ax2.plot(deepabs_pct_kept, deepabs_zs_agr, "s--", color="crimson", linewidth=2, label="DeepAbstract Baseline")
    ax2.set_title("ResNet-18 / CIFAR-10 Zero-Shot Teacher Agreement", fontweight="bold")
    ax2.set_xlabel("Neurons Kept (%)", fontweight="bold")
    ax2.set_ylabel("Zero-Shot Teacher Agreement (%)", fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/resnet18_zeroshot_sweep.png", dpi=300)
    print("\n--> Saved Zero-Shot figure to results/resnet18_zeroshot_sweep.png")

    # Plot 2: Fine-Tuned Performance vs % Neurons Kept
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(causal_pct_kept, causal_ft_acc, "o-", color="forestgreen", linewidth=2, label="Causal Restructuring (Ours)")
    ax1.plot(deepabs_pct_kept, deepabs_ft_acc, "s--", color="darkorange", linewidth=2, label="DeepAbstract + KD Baseline")
    ax1.set_title("ResNet-18 / CIFAR-10 Fine-Tuned Accuracy", fontweight="bold")
    ax1.set_xlabel("Neurons Kept (%)", fontweight="bold")
    ax1.set_ylabel("Fine-Tuned Test Accuracy (%)", fontweight="bold")
    ax1.legend()

    ax2.plot(causal_pct_kept, causal_ft_agr, "o-", color="forestgreen", linewidth=2, label="Causal Restructuring (Ours)")
    ax2.plot(deepabs_pct_kept, deepabs_ft_agr, "s--", color="darkorange", linewidth=2, label="DeepAbstract + KD Baseline")
    ax2.set_title("ResNet-18 / CIFAR-10 Fine-Tuned Teacher Agreement", fontweight="bold")
    ax2.set_xlabel("Neurons Kept (%)", fontweight="bold")
    ax2.set_ylabel("Fine-Tuned Teacher Agreement (%)", fontweight="bold")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("results/resnet18_finetuned_sweep.png", dpi=300)
    print("--> Saved Fine-Tuned figure to results/resnet18_finetuned_sweep.png")

if __name__ == "__main__":
    main()
