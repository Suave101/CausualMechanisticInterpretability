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
from src.utils.AlphaBetaCrown_Verifier.verifier_wrapper import AlphaBetaCrownVerifier

def parse_args():
    parser = argparse.ArgumentParser(description="Inference-Only Causal Rearrangement & Abstraction")
    parser.add_argument("--model", type=str, default="resnet18", choices=["mlp3", "mlp5", "resnet18", "resnet50", "resnet101"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["uci_digits", "cifar10", "cifar100", "imagenet1k"])
    parser.add_argument("--top_n_pct", type=float, default=15.0, help="Percentage of top causal neurons shifted to front layers")
    parser.add_argument("--use_pretrained", action="store_true", default=True, help="Load pretrained weights where applicable")
    parser.add_argument("--force_train", action="store_true", help="Force retraining and overwrite cached weights")
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()

def run_inference_eval(model, test_loader, device):
    """Calculates accuracy over test_loader strictly in inference mode."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return (correct / total) * 100.0

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model} | Dataset: {args.dataset}")

    # 1. Instantiate Data Loaders (Used strictly for inference/profiling)
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

    # 3. Load Cached / Pretrained Weights (Bypasses training if cache exists)
    model = load_or_cache_model(
        model=model,
        model_name=args.model,
        dataset_name=args.dataset,
        device=device,
        train_loader=train_loader,  # Only used if no cache exists
        force_train=args.force_train
    ).to(device)

    # 4. Pure Inference Accuracy Check
    baseline_acc = run_inference_eval(model, test_loader, device)
    print(f"[Inference Output] Baseline Teacher Test Accuracy: {baseline_acc:.2f}%")

    # 5. Causal Profiling using Inference Data
    print("\n[Inference Profiling] Extracting fingerprints and calculating causal scores...")
    all_fps, scores, unit_map = compute_causal_fingerprints_and_scores(model, val_loader, device)

    # 6. Front-Loaded Rearrangement Assignment
    top_indices, rem_indices = rearrange_neurons_front_loaded(scores, top_n_pct=args.top_n_pct)
    print(f"  -> Total Units Analyzed: {len(scores)}")
    print(f"  -> Relocating Top {args.top_n_pct}% ({len(top_indices)} units) to Early Layer.")

    # 7. Verification Interface Pass
    verifier = AlphaBetaCrownVerifier(model)
    sample_x, _ = next(iter(test_loader))
    verif_res = verifier.verify_robustness(sample_x.to(device))
    print(f"[Formal Verification] Result: {verif_res}")

if __name__ == "__main__":
    main()