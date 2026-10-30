# experiments/experiment_restructuring_suite.py
"""
Zero-shot restructuring suite for CIFAR ResNets: DeepAbstract, causal restructuring with and without
least-squares reconstruction (LSQ) and the sensitivity budget, and selection controls that share the
LSQ rebuild. Results are appended to <out_dir>/suite.json after every run, so a restarted job skips
everything already finished. Plots and a markdown table are written at the end.

    python experiments/experiment_restructuring_suite.py --model resnet50 --seed 0
"""
import argparse
import contextlib
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.DataLoader import get_dataset
from src.models.resnet_cifar import ResNet18CIFAR, ResNet50CIFAR
from src.utils.CMI_Method import compute_causal_fingerprints_and_scores
from src.utils import CausalRestructuringMethod as CR
from src.utils.DeepAbstractBaseline import run_deepabstract_baseline

# name, fold_mode, sensitivity budget, reserve share s (fraction of kept units chosen by causal score)
CONFIGS = [
    ("Causal (fold)", "sqrt", False, 0.3),
    ("Causal + budget", "sqrt", True, 0.3),
    ("Causal + LSQ", "lsq", False, 0.3),
    ("Causal + LSQ + budget (s=0.3)", "lsq", True, 0.3),
    ("Causal + LSQ + budget (s=1)", "lsq", True, 1.0),
]
CONTROLS = ["topscore", "kmeans", "random"]
CONTROL_NAMES = {"topscore": "Top-score + LSQ (uniform)", "kmeans": "k-means + LSQ", "random": "Random + LSQ"}

quiet = lambda: contextlib.redirect_stdout(io.StringIO())


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet50", choices=["resnet18", "resnet50"])
    p.add_argument("--dataset", default="cifar10", choices=["cifar10"])
    p.add_argument("--seed", type=int, default=0, help="teacher seed; selects checkpoints/<model>_<dataset>_seedN.pth for N>0")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out_dir", default=None, help="default: results/<model>_seed<N>")
    p.add_argument("--targets", type=float, nargs="+", default=[10.0, 20.0, 30.0, 50.0, 80.0])
    p.add_argument("--calib_samples", type=int, default=2048, help="samples for LSQ statistics")
    p.add_argument("--smoke", action="store_true", help="tiny sample counts and a test subset, for checking the pipeline only")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suffix = f"_seed{args.seed}" if args.seed else ""
    model_id = f"{args.model}_{args.dataset}{suffix}"
    ckpt = args.checkpoint or os.path.join("checkpoints", f"{model_id}.pth")
    out_dir = args.out_dir or os.path.join("results", f"{args.model}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "cache") if args.smoke else "cache"
    t0 = time.time()
    log = lambda *a: print(f"[{time.time()-t0:7.0f}s]", *a, flush=True)

    # Smoke mode shrinks every sample count so the whole pipeline runs in minutes
    profile_samples = 16 if args.smoke else 256
    calib_samples = 128 if args.smoke else args.calib_samples
    sens_kwargs = dict(num_samples=64, recal_batches=2, keep_grid=(0.1, 0.3, 0.5, 1.0)) if args.smoke else {}

    seed_all(0)
    train_loader, val_loader, test_loader, num_classes, _, _ = get_dataset(args.dataset, batch_size=32)
    if args.smoke:
        test_loader = DataLoader(Subset(test_loader.dataset, range(512)), batch_size=32, shuffle=False)

    model_cls = ResNet50CIFAR if args.model == "resnet50" else ResNet18CIFAR
    model = model_cls(num_classes=num_classes, load_pretrained=False)
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"No teacher at {ckpt}; train one with experiments/train_teacher_cifar.py")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device).eval()

    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    teacher_acc = 100.0 * correct / total
    log(f"{model_id}: teacher test acc {teacher_acc:.2f}% ({ckpt})")

    # Resume: keep every finished (config, target) from a previous run
    json_path = os.path.join(out_dir, "suite.json")
    state = {"model": args.model, "seed": args.seed, "teacher_acc": teacher_acc, "results": [], "sensitivity": {}}
    if os.path.exists(json_path):
        prev = json.load(open(json_path))
        state["results"], state["sensitivity"] = prev.get("results", []), prev.get("sensitivity", {})
        log(f"resuming: {len(state['results'])} results already in {json_path}")
    done = {(r["config"], r["target"]) for r in state["results"]}

    def save():
        tmp = json_path + ".tmp"
        json.dump(state, open(tmp, "w"), indent=1)
        os.replace(tmp, json_path)

    def record(config, target, n_kept, n_total, zs):
        # A correct prediction that disagrees with the teacher implies the teacher is wrong
        sane = zs["zs_acc"] - zs["zs_agr"] <= 100.0 - teacher_acc + 1e-6
        if not sane:
            log(f"WARNING: inconsistent acc/agreement for {config} @ {target}%: {zs}")
        state["results"].append({"config": config, "target": target, "pct_kept": 100.0 * n_kept / n_total,
                                 "zs_acc": zs["zs_acc"], "zs_agr": zs["zs_agr"], "sane": sane})
        done.add((config, target))
        save()
        log(f"{config:32s} {target:5.1f}% kept -> acc {zs['zs_acc']:5.2f}  agr {zs['zs_agr']:5.2f}")

    # Causal scores and fingerprints (fp32, cached per teacher)
    log("computing causal scores and fingerprints...")
    fps, scores, unit_map = compute_causal_fingerprints_and_scores(
        model, val_loader, device, max_profile_samples=profile_samples,
        cache_dir=cache_dir, model_id=None if args.smoke else model_id)
    n_units = len(scores)
    log(f"{n_units} units in {len(CR.layer_unit_groups(unit_map))} layers")

    act_stats = None
    def get_act_stats():
        nonlocal act_stats
        if act_stats is None:
            seed_all(0)
            act_stats = CR.collect_activation_stats(model, train_loader, device, max_samples=calib_samples)
        return act_stats

    # DeepAbstract baseline
    for target in args.targets:
        if ("DeepAbstract", target) in done:
            continue
        seed_all(0)
        with quiet():
            da = run_deepabstract_baseline(model, val_loader, test_loader, reduction_rate=1.0 - target / 100.0,
                                           device=device, model_id=model_id, cache_dir=cache_dir, ft_epochs=0,
                                           num_calibration_samples=profile_samples if args.smoke else 54000)
        record("DeepAbstract", target, da["kept_units"], da["total_units"],
               {"zs_acc": da["zs_acc"], "zs_agr": da["zs_agreement"]})

    # Causal restructuring variants
    for name, fold, budget, share in CONFIGS:
        todo = [t for t in args.targets if (name, t) not in done]
        if not todo:
            continue
        stats = get_act_stats() if fold == "lsq" else None
        sens = None
        if budget:
            key = f"{fold}_s{share}"
            if key not in state["sensitivity"]:
                log(f"profiling layer sensitivity for {key}...")
                seed_all(0)
                prof = CR.profile_layer_sensitivity(model, fps, scores, unit_map, train_loader, device,
                                                    reserve_share=share, fold_mode=fold, act_stats=stats, **sens_kwargs)
                state["sensitivity"][key] = {str(l): {str(r): v for r, v in c.items()} for l, c in prof.items()}
                save()
            sens = {int(l): {float(r): v for r, v in c.items()} for l, c in state["sensitivity"][key].items()}
        for target in todo:
            seed_all(0)
            reserve = share * target
            bg_rr = 1.0 - (target - reserve) / (100.0 - reserve) if share < 1.0 else 1.0
            with quiet():
                _, kept, _, zs = CR.perform_causal_restructuring(
                    teacher_model=model, val_loader=val_loader, test_loader=test_loader,
                    top_n_pct=reserve, reduction_rate=bg_rr, causal_weight=0.5, device=device,
                    all_fps=fps, ablation_scores=scores, unit_map=unit_map, fold_mode=fold,
                    budget_mode="sensitivity" if budget else "uniform", calib_loader=train_loader,
                    act_stats=stats, layer_sensitivity=sens)
            record(name, target, len(kept), n_units, zs)

    # Selection controls: same LSQ rebuild and uniform per-layer ratio, different choice of kept units
    groups = CR.layer_unit_groups(unit_map)
    for mode in CONTROLS:
        name = CONTROL_NAMES[mode]
        rng = np.random.default_rng(args.seed)
        for target in args.targets:
            if (name, target) in done:
                continue
            cmap, kept = {}, []
            for l_idx, idx in groups.items():
                nl = len(idx)
                k = max(1, int(round(target / 100.0 * nl)))
                if mode == "random":
                    sel = rng.choice(nl, k, replace=False)
                elif mode == "topscore":
                    sel = np.argsort(scores[idx])[::-1][:k]
                else:
                    f = fps[idx]
                    km = KMeans(n_clusters=k, random_state=42, n_init=5).fit(f)
                    sel = [np.where(km.labels_ == c)[0][np.argmin(np.linalg.norm(f[km.labels_ == c] - km.cluster_centers_[c], axis=1))]
                           for c in range(k) if np.any(km.labels_ == c)]
                cmap[l_idx] = [(int(s), [int(s)]) for s in sel]
                kept += [idx[int(s)] for s in sel]
            sub = CR.build_restructured_model(model, cmap, "lsq", get_act_stats(), device)
            with quiet():
                zs = CR.evaluate_masked_model(sub, model, test_loader, kept, n_units, device, val_loader=val_loader)
            record(name, target, len(kept), n_units, zs)

    write_report(state, out_dir, args.model)
    log(f"done; results in {out_dir}")


def write_report(state, out_dir, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    markers = ["o", "s", "^", "D", "v"]
    plt.rcParams.update({
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface,
        "axes.edgecolor": muted, "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2,
        "text.color": ink, "font.size": 10, "axes.titleweight": "bold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": grid, "legend.frameon": False,
    })
    title = {"resnet50": "ResNet-50", "resnet18": "ResNet-18"}[model_name]
    rows = state["results"]

    def curve(config, metric):
        rs = sorted([r for r in rows if r["config"] == config], key=lambda r: r["target"])
        return [r["target"] for r in rs], [r[metric] for r in rs]

    def panel(ax, configs, metric, ref, ref_label):
        ax.axhline(ref, color=muted, lw=1.2, ls="--", zorder=1)
        ax.text(1, ref + 0.8, ref_label, color=ink2, fontsize=8.5, ha="left", va="bottom")
        for i, cfg in enumerate(configs):
            x, y = curve(cfg, metric)
            if x:
                ax.plot(x, y, color=colors[i], lw=2, marker=markers[i], ms=6.5, mec=surface, mew=1.5, label=cfg, zorder=3)
        ax.set_xlabel("Units kept (%)"); ax.set_ylim(0, 106); ax.set_xlim(0, 84)
        ax.set_yticks(range(0, 101, 20)); ax.set_xticks(sorted({r["target"] for r in rows}))

    main_cfgs = ["Causal + LSQ + budget (s=1)", "DeepAbstract", "Causal (fold)", "Causal + LSQ", "Causal + LSQ + budget (s=0.3)"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panel(axes[0], main_cfgs, "zs_acc", state["teacher_acc"], f"teacher {state['teacher_acc']:.1f}%")
    axes[0].set_ylabel("Zero-shot test accuracy (%)"); axes[0].set_title("Accuracy", loc="left")
    panel(axes[1], main_cfgs, "zs_agr", 100.0, "teacher (100%)")
    axes[1].set_ylabel("Agreement with teacher (%)"); axes[1].set_title("Agreement", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=9)
    fig.suptitle(f"{title} · CIFAR-10 (teacher seed {state['seed']}): zero-shot restructuring", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(os.path.join(out_dir, f"{model_name}_restructuring.png"), dpi=160)
    plt.close(fig)

    ctrl_cfgs = ["Causal + LSQ + budget (s=1)"] + [CONTROL_NAMES[m] for m in CONTROLS]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    panel(ax, ctrl_cfgs, "zs_acc", state["teacher_acc"], f"teacher {state['teacher_acc']:.1f}%")
    ax.set_ylabel("Zero-shot test accuracy (%)"); ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{title}: which units to keep (all use LSQ reconstruction)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{model_name}_selection_controls.png"), dpi=160)
    plt.close(fig)

    targets = sorted({r["target"] for r in rows})
    lines = [f"# {title} / CIFAR-10, teacher seed {state['seed']} (teacher acc {state['teacher_acc']:.2f}%)", "",
             "Zero-shot accuracy / agreement (%).", "",
             "| Config | " + " | ".join(f"{t:g}%" for t in targets) + " |", "|---|" + "---|" * len(targets)]
    for cfg in dict.fromkeys(r["config"] for r in rows):
        cells = {r["target"]: f"{r['zs_acc']:.1f} / {r['zs_agr']:.1f}" for r in rows if r["config"] == cfg}
        lines.append(f"| {cfg} | " + " | ".join(cells.get(t, "") for t in targets) + " |")
    open(os.path.join(out_dir, f"{model_name}_results.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
