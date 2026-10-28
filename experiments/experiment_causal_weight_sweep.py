# experiments/experiment_causal_weight_sweep.py

import os
import re
import subprocess
import sys
import matplotlib.pyplot as plt
import numpy as np


def run_experiment_command(cmd_args):
    """Executes main.py as a subprocess and streams output live."""
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
    """Parses accuracy and agreement metrics from standard console output."""
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
    
    # Hyperparameter search values for Alpha (Causal Weight)
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    # Capacity targets to test across
    target_capacity_pcts = [30.0, 50.0, 80.0]
    
    results = {cap: {"zs_acc": [], "zs_agr": [], "ft_acc": [], "ft_agr": []} for cap in target_capacity_pcts}

    print(f"===========================================================")
    print(f" Causal Weight (Alpha) Ablation Sweep: {model.upper()} on {dataset}")
    print(f" Testing Alphas: {alpha_values} across Capacities: {target_capacity_pcts}%")
    print(f"===========================================================\n")

    for target_pct in target_capacity_pcts:
        # Calculate Causal Restructuring parameters for target total capacity
        causal_reserve = 0.3 * target_pct
        bg_total_pct = 100.0 - causal_reserve
        bg_kept_pct = target_pct - causal_reserve
        causal_bg_rr = 1.0 - (bg_kept_pct / bg_total_pct)

        for alpha in alpha_values:
            print(f"\n---> Capacity = {target_pct}% | Alpha (Causal Weight) = {alpha}")
            
            cmd_args = [
                "--method", "causal",
                "--model", model,
                "--dataset", dataset,
                "--batch_size", str(batch_size),
                "--top_n_pct", f"{causal_reserve:.4f}",
                "--reduction_rate", f"{causal_bg_rr:.4f}",
                "--causal_weight", str(alpha),
                "--ft_epochs", str(ft_epochs)
            ]
            
            out = run_experiment_command(cmd_args)
            m = parse_metrics(out)
            
            results[target_pct]["zs_acc"].append(m.get("zs_acc", np.nan))
            results[target_pct]["zs_agr"].append(m.get("zs_agr", np.nan))
            results[target_pct]["ft_acc"].append(m.get("ft_acc", np.nan))
            results[target_pct]["ft_agr"].append(m.get("ft_agr", np.nan))

    # --- Plotting Results ---
    os.makedirs("results", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    colors = {30.0: "#1f77b4", 50.0: "#2ca02c", 80.0: "#ff7f0e"}

    for cap in target_capacity_pcts:
        c = colors[cap]
        ax1.plot(alpha_values, results[cap]["zs_acc"], "o-", color=c, linewidth=2, label=f"Cap {cap}%")
        ax2.plot(alpha_values, results[cap]["zs_agr"], "o-", color=c, linewidth=2, label=f"Cap {cap}%")
        ax3.plot(alpha_values, results[cap]["ft_acc"], "s--", color=c, linewidth=2, label=f"Cap {cap}%")
        ax4.plot(alpha_values, results[cap]["ft_agr"], "s--", color=c, linewidth=2, label=f"Cap {cap}%")

    ax1.set_title("Zero-Shot Accuracy vs. Causal Weight (Alpha)", fontweight="bold")
    ax1.set_xlabel("Causal Weight (Alpha)", fontweight="bold")
    ax1.set_ylabel("Zero-Shot Test Accuracy (%)", fontweight="bold")
    ax1.legend()

    ax2.set_title("Zero-Shot Teacher Agreement vs. Causal Weight (Alpha)", fontweight="bold")
    ax2.set_xlabel("Causal Weight (Alpha)", fontweight="bold")
    ax2.set_ylabel("Zero-Shot Agreement (%)", fontweight="bold")
    ax2.legend()

    ax3.set_title("Fine-Tuned Accuracy vs. Causal Weight (Alpha)", fontweight="bold")
    ax3.set_xlabel("Causal Weight (Alpha)", fontweight="bold")
    ax3.set_ylabel("Fine-Tuned Test Accuracy (%)", fontweight="bold")
    ax3.legend()

    ax4.set_title("Fine-Tuned Teacher Agreement vs. Causal Weight (Alpha)", fontweight="bold")
    ax4.set_xlabel("Causal Weight (Alpha)", fontweight="bold")
    ax4.set_ylabel("Fine-Tuned Agreement (%)", fontweight="bold")
    ax4.legend()

    plt.tight_layout()
    fig_path = os.path.join("results", "mlp_causal_weight_ablation.png")
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    print("\n===========================================================")
    print(f" ABLATION SWEEP COMPLETE")
    print(f" Saved Figure -> {fig_path}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
