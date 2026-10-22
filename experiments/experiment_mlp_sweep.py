import os
import sys
import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np


def run_experiment_command(cmd_args):
    """Runs main.py as a subprocess and captures STDOUT."""
    cmd = [sys.executable, "main.py"] + cmd_args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {result.stderr}")
        return None
    return result.stdout


def parse_causal_metrics(output_text):
    metrics = {}
    zs_acc_match = re.search(r"Zero-Shot Accuracy:\s+([0-9\.]+)%", output_text)
    zs_agr_match = re.search(r"Zero-Shot Teacher Agreement:\s+([0-9\.]+)%", output_text)
    ft_acc_match = re.search(r"Fine-Tuned Accuracy:\s+([0-9\.]+)%", output_text)
    ft_agr_match = re.search(r"Fine-Tuned Teacher Agreement:\s+([0-9\.]+)%", output_text)
    
    if zs_acc_match: metrics["zs_acc"] = float(zs_acc_match.group(1))
    if zs_agr_match: metrics["zs_agr"] = float(zs_agr_match.group(1))
    if ft_acc_match: metrics["ft_acc"] = float(ft_acc_match.group(1))
    if ft_agr_match: metrics["ft_agr"] = float(ft_agr_match.group(1))
    return metrics


def parse_deepabstract_metrics(output_text):
    metrics = {}
    zs_acc_match = re.search(r"Zero-Shot Accuracy:\s+([0-9\.]+)%", output_text)
    zs_agr_match = re.search(r"Zero-Shot Agreement:\s+([0-9\.]+)%", output_text)
    ft_acc_match = re.search(r"Fine-Tuned Accuracy:\s+([0-9\.]+)%", output_text)
    ft_agr_match = re.search(r"Fine-Tuned Teacher Agreement:\s+([0-9\.]+)%", output_text)
    
    if zs_acc_match: metrics["zs_acc"] = float(zs_acc_match.group(1))
    if zs_agr_match: metrics["zs_agr"] = float(zs_agr_match.group(1))
    if ft_acc_match: metrics["ft_acc"] = float(ft_acc_match.group(1))
    if ft_agr_match: metrics["ft_agr"] = float(ft_agr_match.group(1))
    return metrics


def main():
    model = "mlp3"
    dataset = "uci_digits"
    batch_size = 32
    ft_epochs = 5
    
    top_n_pct_range = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0]
    
    causal_zs_acc, causal_zs_agr = [], []
    causal_ft_acc, causal_ft_agr = [], []
    da_zs_acc, da_zs_agr = [], []
    da_ft_acc, da_ft_agr = [], []

    print(f"===========================================================")
    print(f" Starting Scientific Sweep: {model.upper()} on {dataset}")
    print(f"===========================================================\n")

    for top_pct in top_n_pct_range:
        print(f"\n--- Testing Reserve Budget = {top_pct}% ---")
        
        # 1. Run Proposed Causal Restructuring Framework
        causal_args = [
            "--method", "causal",
            "--model", model,
            "--dataset", dataset,
            "--batch_size", str(batch_size),
            "--top_n_pct", str(top_pct),
            "--ft_epochs", str(ft_epochs)
        ]
        causal_out = run_experiment_command(causal_args)
        c_metrics = parse_causal_metrics(causal_out) if causal_out else {}
        
        causal_zs_acc.append(c_metrics.get("zs_acc", np.nan))
        causal_zs_agr.append(c_metrics.get("zs_agr", np.nan))
        causal_ft_acc.append(c_metrics.get("ft_acc", np.nan))
        causal_ft_agr.append(c_metrics.get("ft_agr", np.nan))

        # 2. Run DeepAbstract Baseline with same FT budget
        reduction_rate = round(1.0 - ((top_pct / 100.0) + (1.0 - (top_pct / 100.0)) * 0.5), 3)
        da_args = [
            "--method", "deepabstract",
            "--model", model,
            "--dataset", dataset,
            "--batch_size", str(batch_size),
            "--reduction_rate", str(reduction_rate),
            "--ft_epochs", str(ft_epochs)
        ]
        da_out = run_experiment_command(da_args)
        d_metrics = parse_deepabstract_metrics(da_out) if da_out else {}
        
        da_zs_acc.append(d_metrics.get("zs_acc", np.nan))
        da_zs_agr.append(d_metrics.get("zs_agr", np.nan))
        da_ft_acc.append(d_metrics.get("ft_acc", np.nan))
        da_ft_agr.append(d_metrics.get("ft_agr", np.nan))

    os.makedirs("results", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -----------------------------------------------------------------
    # GRAPH 1: ZERO-SHOT COMPARISON (No Gradient Updates)
    # -----------------------------------------------------------------
    fig1, (ax1_1, ax1_2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1_1.plot(top_n_pct_range, causal_zs_acc, "o-", color="#1f77b4", linewidth=2, label="Causal Restructuring (Ours)")
    ax1_1.plot(top_n_pct_range, da_zs_acc, "s--", color="#d62728", linewidth=2, label="DeepAbstract Baseline")
    ax1_1.set_xlabel("Reserve Budget (--top_n_pct %)", fontsize=11)
    ax1_1.set_ylabel("Zero-Shot Test Accuracy (%)", fontsize=11)
    ax1_1.set_title("Zero-Shot Classification Accuracy", fontsize=12, fontweight="bold")
    ax1_1.legend(loc="lower right")
    ax1_1.grid(True, alpha=0.3)

    ax1_2.plot(top_n_pct_range, causal_zs_agr, "o-", color="#1f77b4", linewidth=2, label="Causal Restructuring (Ours)")
    ax1_2.plot(top_n_pct_range, da_zs_agr, "s--", color="#d62728", linewidth=2, label="DeepAbstract Baseline")
    ax1_2.set_xlabel("Reserve Budget (--top_n_pct %)", fontsize=11)
    ax1_2.set_ylabel("Zero-Shot Teacher Agreement (%)", fontsize=11)
    ax1_2.set_title("Zero-Shot Functional Agreement", fontsize=12, fontweight="bold")
    ax1_2.legend(loc="lower right")
    ax1_2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = os.path.join("results", "mlp_zero_shot_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # -----------------------------------------------------------------
    # GRAPH 2: FINE-TUNED COMPARISON (KD + EMA Fine-Tuning)
    # -----------------------------------------------------------------
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax2_1.plot(top_n_pct_range, causal_ft_acc, "o-", color="#2ca02c", linewidth=2, label="Causal Restructuring (Ours)")
    ax2_1.plot(top_n_pct_range, da_ft_acc, "s--", color="#ff7f0e", linewidth=2, label="DeepAbstract + KD Baseline")
    ax2_1.set_xlabel("Reserve Budget (--top_n_pct %)", fontsize=11)
    ax2_1.set_ylabel("Fine-Tuned Test Accuracy (%)", fontsize=11)
    ax2_1.set_title("Post Fine-Tuning Classification Accuracy", fontsize=12, fontweight="bold")
    ax2_1.legend(loc="lower right")
    ax2_1.grid(True, alpha=0.3)

    ax2_2.plot(top_n_pct_range, causal_ft_agr, "o-", color="#2ca02c", linewidth=2, label="Causal Restructuring (Ours)")
    ax2_2.plot(top_n_pct_range, da_ft_agr, "s--", color="#ff7f0e", linewidth=2, label="DeepAbstract + KD Baseline")
    ax2_2.set_xlabel("Reserve Budget (--top_n_pct %)", fontsize=11)
    ax2_2.set_ylabel("Fine-Tuned Teacher Agreement (%)", fontsize=11)
    ax2_2.set_title("Post Fine-Tuning Functional Agreement", fontsize=12, fontweight="bold")
    ax2_2.legend(loc="lower right")
    ax2_2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig2_path = os.path.join("results", "mlp_fine_tuned_comparison.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close(fig2)

    print("\n===========================================================")
    print(" SCIENTIFIC SWEEP COMPLETE")
    print(f" Saved Graph 1 (Zero-Shot)  -> {fig1_path}")
    print(f" Saved Graph 2 (Fine-Tuned) -> {fig2_path}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
