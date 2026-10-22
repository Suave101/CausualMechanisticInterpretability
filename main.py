# main.py
import argparse
import torch

from src.utils.DataLoader import get_dataset
from src.utils.checkpoint_utils import load_or_cache_model
from src.models.mlp import build_3layer_mlp, build_5layer_mlp
from src.models.resnet_cifar import ResNet18CIFAR
from src.models.resnet_imagenet import ResNetImageNet
from src.utils.CMI_Method import (
    compute_causal_fingerprints_and_scores,
    rearrange_neurons_front_loaded
)
from src.utils.DeepAbstractMethod import construct_and_evaluate_submodel
from src.utils.AlphaBetaCrown_Verifier.verifier_wrapper import AlphaBetaCrownVerifier

def parse_args():
    parser = argparse.ArgumentParser(description="Inference-Only Causal Rearrangement & Abstraction")
    parser.add_argument("--model", type=str, default="resnet18", choices=["mlp3", "mlp5", "resnet18", "resnet50", "resnet101"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["uci_digits", "cifar10", "cifar100", "imagenet1k"])
    parser.add_argument("--top_n_pct", type=float, default=15.0, help="Percentage of top causal neurons shifted to front layers")
    parser.add_argument("--use_pretrained", action="store_true", default=True)
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ft_epochs", type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | Dataset: {args.dataset}")

    # 1. Instantiate Data Loaders
    train_loader, val_loader, test_loader, num_classes, in_dim, _ = get_dataset(
        args.dataset, batch_size=args.batch_size
    )

    # 2. Instantiate Model
    if args.model == "mlp3":
        model = build_3layer_mlp(in_dim=in_dim, num_classes=num_classes)
    elif args.model == "mlp5":
        model = build_5layer_mlp(in_dim=in_dim, num_classes=num_classes)
    elif args.model == "resnet18" and "cifar" in args.dataset:
        model = ResNet18CIFAR(num_classes=num_classes, load_pretrained=args.use_pretrained)
    elif args.model in ["resnet18", "resnet50", "resnet101"]:
        model = ResNetImageNet(variant=args.model, pretrained=args.use_pretrained)
    else:
        raise ValueError(f"Unsupported model-dataset combination: {args.model} on {args.dataset}")

    # 3. Load Cached / Pretrained Weights
    model = load_or_cache_model(
        model=model,
        model_name=args.model,
        dataset_name=args.dataset,
        device=device,
        train_loader=train_loader,
        force_train=args.force_train
    ).to(device)

    # 4. Extract Causal Scores & Fingerprints
    print("\n[Inference Profiling] Extracting fingerprints and calculating causal scores...")
    all_fps, scores, unit_map = compute_causal_fingerprints_and_scores(model, val_loader, device)

    # 5. Front-Loaded Rearrangement Partition
    top_indices, rem_indices = rearrange_neurons_front_loaded(scores, top_n_pct=args.top_n_pct)

    # 6. Physical Restructuring, Zero-Shot, and KD + EMA Fine-Tuning
    results = construct_and_evaluate_submodel(
        teacher_model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        all_fps=all_fps,
        ablation_scores=scores,
        top_indices=top_indices,
        rem_indices=rem_indices,
        device=device,
        ft_epochs=args.ft_epochs
    )

    # 7. Verification Interface Pass
    verifier = AlphaBetaCrownVerifier(model)
    sample_x, _ = next(iter(test_loader))
    verif_res = verifier.verify_robustness(sample_x.to(device))
    print(f"\n[Formal Verification] Result: {verif_res}")

if __name__ == "__main__":
    main()
