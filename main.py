# main.py
import argparse
import os
import torch

from src.utils.DataLoader import get_dataset
from src.utils.checkpoint_utils import load_or_cache_model
from src.models.mlp import build_3layer_mlp, build_5layer_mlp
from src.models.resnet_cifar import ResNet18CIFAR, ResNet50CIFAR
from src.models.resnet_imagenet import ResNetImageNet

# Causal Restructuring Framework Modules
from src.utils.CMI_Method import (
    compute_causal_fingerprints_and_scores,
    rearrange_neurons_front_loaded
)
from src.utils.CausalRestructuringMethod import perform_causal_restructuring
from src.utils.FineTuningMethod import fine_tune_submodel

# DeepAbstract Baseline Module (Ashok et al., 2020)
from src.utils.DeepAbstractBaseline import run_deepabstract_baseline

# Formal Robustness Verifier
from src.utils.AlphaBetaCrown_Verifier.verifier_wrapper import AlphaBetaCrownVerifier


def parse_args():
    parser = argparse.ArgumentParser(description="Neural Network Restructuring & Abstraction Framework")
    
    # Method Selection
    parser.add_argument(
        "--method",
        type=str,
        default="causal",
        choices=["causal", "deepabstract", "both"],
        help="Method to execute: 'causal', 'deepabstract', or 'both'"
    )
    
    # Model & Dataset Options
    parser.add_argument("--model", type=str, default="mlp3", choices=["mlp3", "mlp5", "resnet18", "resnet50", "resnet101"])
    parser.add_argument("--dataset", type=str, default="uci_digits", choices=["uci_digits", "cifar10", "cifar100", "imagenet1k"])
    
    # Causal Framework Parameters
    parser.add_argument("--top_n_pct", type=float, default=15.0, help="Percentage of top causal neurons/channels reserved")
    parser.add_argument("--causal_weight", type=float, default=0.5, help="Weight alpha balance: 1.0 = pure causal score, 0.0 = pure fingerprint distance")
    
    # DeepAbstract Baseline Parameters
    parser.add_argument("--reduction_rate", type=float, default=0.425, help="Target channel reduction ratio for DeepAbstract")
    
    # Setup & Execution Parameters
    parser.add_argument("--use_pretrained", action="store_true", default=True)
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ft_epochs", type=int, default=5)
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | Dataset: {args.dataset} | Method: {args.method.upper()} | Causal Weight (Alpha): {args.causal_weight}")

    # Instantiate Data Loaders
    train_loader, val_loader, test_loader, num_classes, in_dim, _ = get_dataset(
        args.dataset, batch_size=args.batch_size
    )

    # Instantiate Base Model Architecture
    if args.model == "mlp3":
        model = build_3layer_mlp(in_dim=in_dim, num_classes=num_classes)
    elif args.model == "mlp5":
        model = build_5layer_mlp(in_dim=in_dim, num_classes=num_classes)
    elif args.model == "resnet18" and "cifar" in args.dataset:
        model = ResNet18CIFAR(num_classes=num_classes, load_pretrained=args.use_pretrained)
    elif args.model == "resnet50" and "cifar" in args.dataset:
        model = ResNet50CIFAR(num_classes=num_classes, load_pretrained=args.use_pretrained)
    elif args.model in ["resnet18", "resnet50", "resnet101"]:
        model = ResNetImageNet(variant=args.model, pretrained=args.use_pretrained)

    # Load Cached or Pre-Trained Teacher Weights
    model = load_or_cache_model(
        model=model,
        model_name=args.model,
        dataset_name=args.dataset,
        device=device,
        train_loader=train_loader,
        force_train=args.force_train
    ).to(device)

    # Method Execution Pipeline
    if args.method in ["causal", "both"]:
        print("\n=======================================================")
        print(" RUNNING: Causal Restructuring & Compression Framework")
        print("=======================================================")
        
        # Causal Fingerprinting & Score Profiling
        model_id_str = f"{args.model}_{args.dataset}"
        all_fps, scores, unit_map = compute_causal_fingerprints_and_scores(
            model, val_loader, device, model_id=model_id_str
        )
    
        # Causal Restructuring & Compression
        sub_model, sorted_kept, mask_vector, zs_metrics = perform_causal_restructuring(
            teacher_model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            top_n_pct=args.top_n_pct,
            reduction_rate=args.reduction_rate,
            causal_weight=args.causal_weight,
            all_fps=all_fps,
            ablation_scores=scores,
            unit_map=unit_map,
            device=device,
            model_id=model_id_str
        )

        # Fine-Tuning via Knowledge Distillation (KD) + EMA
        ft_metrics = fine_tune_submodel(
            sub_model=sub_model,
            teacher_model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            mask_vector=mask_vector,
            ft_epochs=args.ft_epochs
        )

    if args.method in ["deepabstract", "both"]:
        print("\n=======================================================")
        print(" RUNNING: DeepAbstract Baseline (Ashok et al., ATVA 2020)")
        print("=======================================================")
        
        model_id_str = f"{args.model}_{args.dataset}"
        da_results = run_deepabstract_baseline(
            teacher_model=model,
            val_loader=val_loader,
            test_loader=test_loader,
            train_loader=train_loader,
            reduction_rate=args.reduction_rate,
            cluster_method="kmeans",
            device=device,
            model_id=model_id_str,
            ft_epochs=args.ft_epochs
        )

    # Formal Safety Verification Pass
    verifier = AlphaBetaCrownVerifier(model)
    sample_x, _ = next(iter(test_loader))
    verif_res = verifier.verify_robustness(sample_x.to(device))
    print(f"\n[Formal Verification] Result: {verif_res}")


if __name__ == "__main__":
    main()
