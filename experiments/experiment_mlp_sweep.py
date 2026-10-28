# experiments/experiment_mlp_sweep.py

import os
import re
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np


def run_experiment_command(cmd_args):
    """Executes main.py and streams console output live in real time while capturing stdout."""
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


def parse_metrics(output_text):
    """Extracts accuracy, agreement, and actual percentage of units retained."""
    if not output_text:
        return {}
    metrics = {}
    
    retained_match = re.search(r"Units Retained:\s*\d+\s*/\s*\d+\s*\(([\d\.]+)%\)", output_text)
    if retained_match:
        metrics["pct_kept"] = float(retained_match.group(1))

    zs_acc_match = re.search(r"Zero-Shot Accuracy:\s*([\d\.]+)%", output_text)
    if zs_acc_match:
        metrics["zs_acc"] = float(zs_acc_match.group(1))

    zs_agr_match = re.search(r"Zero-Shot (?:Teacher )?Agreement:\s*([\d\.]+)%", output_text)
    if zs_agr_match:
        metrics["zs_agr"] = float(zs_agr_match.group(1))

    ft_acc_match = re.search(r"Fine-Tuned Accuracy:\s*([\d\.]+)%", output_text)
    if ft_acc_match:
        metrics["ft_acc"] = float(ft_acc_match.group(1))

    ft_agr_match = re.search(r"Fine-Tuned Teacher Agreement:\s*([\d\.]+)%", output_text)
    if ft_agr_match:
        metrics["ft_agr"] = float(ft_agr_match.group(1))

    return metrics


def main():
    model = "mlp3"
    dataset = "uci_digits"
    batch_size = 32
    ft_epochs = 5
    
    # Standardized Total Capacity Targets (%)
    target_kept_pcts = [10.0, 30.0, 50.0, 80.0, 100.0]
    
    causal_pct_kept, causal_zs_acc, causal_zs_agr, causal_ft_acc, causal_ft_agr = [], [], [], [], []
    da_pct_kept, da_zs_acc, da_zs_agr, da_ft_acc, da_ft_agr = [], [], [], [], []

    print(f"===========================================================")
    print(f" Starting Scientific Sweep: {model.upper()} on {dataset}")
    print(f"===========================================================\n")

    for target_pct in target_kept_pcts:
        print(f"\n===========================================================")
        print(f" Testing Target Total Capacity = {target_pct}% Neurons Kept")
        print(f"===========================================================")
        
        # Calculate Causal Restructuring parameters for target total capacity
        if target_pct >= 100.0:
            causal_reserve = 100.0
            causal_bg_rr = 0.0
        else:
            causal_reserve = 0.3 * target_pct
            bg_total_pct = 100.0 - causal_reserve
            bg_kept_pct = target_pct - causal_reserve
            causal_bg_rr = 1.0 - (bg_kept_pct / bg_total_pct)

        # 1. Run Causal Restructuring (Ours)
        causal_args = [
            "--method", "causal",
            "--model", model,
            "--dataset", dataset,
            "--batch_size", str(batch_size),
            "--top_n_pct", f"{causal_reserve:.4f}",
            "--reduction_rate", f"{causal_bg_rr:.4f}",
            "--ft_epochs", str(ft_epochs)
        ]
        causal_out = run_experiment_command(causal_args)
        c_metrics = parse_metrics(causal_out)
        
        causal_pct_kept.append(c_metrics.get("pct_kept", target_pct))
        causal_zs_acc.append(c_metrics.get("zs_acc", np.nan))
        causal_zs_agr.append(c_metrics.get("zs_agr", np.nan))
        causal_ft_acc.append(c_metrics.get("ft_acc", np.nan))
        causal_ft_agr.append(c_metrics.get("ft_agr", np.nan))

        # 2. Run DeepAbstract Baseline
        deepabs_rr = 1.0 - (target_pct / 100.0)
        da_args = [
            "--method", "deepabstract",
            "--model", model,
            "--dataset", dataset,
            "--batch_size", str(batch_size),
            "--reduction_rate", f"{deepabs_rr:.4f}",
            "--ft_epochs", str(ft_epochs)
        ]
        da_out = run_experiment_command(da_args)
        d_metrics = parse_metrics(da_out)
        
        da_pct_kept.append(d_metrics.get("pct_kept", target_pct))
        da_zs_acc.append(d_metrics.get("zs_acc", np.nan))
        da_zs_agr.append(d_metrics.get("zs_agr", np.nan))
        da_ft_acc.append(d_metrics.get("ft_acc", np.nan))
        da_ft_agr.append(d_metrics.get("ft_agr", np.nan))

    os.makedirs("results", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # -----------------------------------------------------------------
    # GRAPH 1: ZERO-SHOT COMPARISON
    # -----------------------------------------------------------------
    fig1, (ax1_1, ax1_2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1_1.plot(causal_pct_kept, causal_zs_acc, "o-", color="#1f77b4", linewidth=2, label="Causal Restructuring (Ours)")
    ax1_1.plot(da_pct_kept, da_zs_acc, "s--", color="#d62728", linewidth=2, label="DeepAbstract Baseline")
    ax1_1.set_xlabel("Neurons Kept (%)", fontsize=11, fontweight="bold")
    ax1_1.set_ylabel("Zero-Shot Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax1_1.set_title("Zero-Shot Classification Accuracy", fontsize=12, fontweight="bold")
    ax1_1.legend(loc="lower right")
    ax1_1.grid(True, alpha=0.3)

    ax1_2.plot(causal_pct_kept, causal_zs_agr, "o-", color="#1f77b4", linewidth=2, label="Causal Restructuring (Ours)")
    ax1_2.plot(da_pct_kept, da_zs_agr, "s--", color="#d62728", linewidth=2, label="DeepAbstract Baseline")
    ax1_2.set_xlabel("Neurons Kept (%)", fontsize=11, fontweight="bold")
    ax1_2.set_ylabel("Zero-Shot Teacher Agreement (%)", fontsize=11, fontweight="bold")
    ax1_2.set_title("Zero-Shot Functional Agreement", fontsize=12, fontweight="bold")
    ax1_2.legend(loc="lower right")
    ax1_2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = os.path.join("results", "mlp_zero_shot_comparison.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close(fig1)

    # -----------------------------------------------------------------
    # GRAPH 2: FINE-TUNED COMPARISON
    # -----------------------------------------------------------------
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax2_1.plot(causal_pct_kept, causal_ft_acc, "o-", color="#2ca02c", linewidth=2, label="Causal Restructuring (Ours)")
    ax2_1.plot(da_pct_kept, da_ft_acc, "s--", color="#ff7f0e", linewidth=2, label="DeepAbstract + KD Baseline")
    ax2_1.set_xlabel("Neurons Kept (%)", fontsize=11, fontweight="bold")
    ax2_1.set_ylabel("Fine-Tuned Test Accuracy (%)", fontsize=11, fontweight="bold")
    ax2_1.set_title("Post Fine-Tuning Classification Accuracy", fontsize=12, fontweight="bold")
    ax2_1.legend(loc="lower right")
    ax2_1.grid(True, alpha=0.3)

    ax2_2.plot(causal_pct_kept, causal_ft_agr, "o-", color="#2ca02c", linewidth=2, label="Causal Restructuring (Ours)")
    ax2_2.plot(da_pct_kept, da_ft_agr, "s--", color="#ff7f0e", linewidth=2, label="DeepAbstract + KD Baseline")
    ax2_2.set_xlabel("Neurons Kept (%)", fontsize=11, fontweight="bold")
    ax2_2.set_ylabel("Fine-Tuned Teacher Agreement (%)", fontsize=11, fontweight="bold")
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
