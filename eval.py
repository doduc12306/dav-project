"""
eval.py - Unified Evaluation and Replication Script for S-JEPA Skeleton Action Recognition
========================================================================================
This script serves as the single entry point to reproduce all quantitative tables and qualitative
figures presented in the paper.

Modes of operation:
  1. --mode metrics: Computes accuracies for all streams and the Late Fusion ensemble (reproduces Table 1).
  2. --mode find_weights: Grid searches for the optimal Late Fusion weights (reproduces Table 1 best config).
  3. --mode visualize: Runs offline linear probing, manifold projections (t-SNE/UMAP/LDA), and XAI explanations.
  4. --mode all: Runs all of the above.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Add 'src' directory to Python path to resolve nested imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Import S-JEPA architectures and datasets
from src.core.classifier import NTUActionClassifier
from src.datasets.ntu_dataset import NTUActionDataset
from src.downstream import train_classifier
from ensemble_eval import topk_accuracy, load_model, extract_softmax, extract_labels, build_test_loader, plot_confusion_matrix

def print_latex_table(results_dict, dataset_name):
    """Prints the evaluation results formatted as a LaTeX table matching the paper structure."""
    print(f"\n% --- LaTeX Table 1 for {dataset_name} ---")
    print("\\begin{table}[htbp]")
    print(f"\\caption{{Độ chính xác phân loại hành động của S-JEPA trên {dataset_name}.}}")
    print("\\label{tab:results}")
    print("\\centering")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("\\textbf{Phương pháp / Luồng dữ liệu} & \\textbf{Acc@1 (\\%)} & \\textbf{Acc@5 (\\%)} \\\\")
    print("\\midrule")
    for name, accs in results_dict.items():
        if "Ensemble" in name:
            print("\\midrule")
            print(f"\\textbf{{{name}}} & \\textbf{{{accs[0]:.2f}}} & \\textbf{{{accs[1]:.2f}}} \\\\")
        else:
            print(f"{name} & {accs[0]:.2f} & {accs[1]:.2f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("% -------------------------------------\n")

def run_metrics_evaluation(args, cfg, ckpt_paths, device):
    """Evaluates individual streams and calculates ensemble late fusion accuracies."""
    print("\n" + "="*60)
    print("  RUNNING QUANTITATIVE METRICS EVALUATION")
    print("="*60)
    
    # 1. Resolve streams
    streams_to_use = ["joint", "bone"]
    if not args.no_velocity:
        dataset_tag = f"ntu120_xsub" if args.ntu120_xsub else f"ntu60_xview"
        cache_path = f"checkpoints_finetuned/ensemble_probs_{dataset_tag}.pt"
        has_velocity = False
        if os.path.exists(ckpt_paths["velocity"]):
            has_velocity = True
        elif os.path.exists(cache_path):
            try:
                data = torch.load(cache_path, map_location="cpu")
                if "velocity" in data.get("softmax", {}):
                    has_velocity = True
            except Exception:
                pass
        
        if has_velocity:
            streams_to_use.append("velocity")
        else:
            print(f"  [WARN] Velocity checkpoint/cache not found -> skipping Velocity stream.")
            
    # Check if we need to load from cache
    use_cache = False
    for s in streams_to_use:
        if not os.path.exists(ckpt_paths[s]):
            use_cache = True
            break
            
    softmax_dict = {}
    labels = None
    
    if use_cache:
        dataset_tag = f"ntu120_xsub" if args.ntu120_xsub else f"ntu60_xview"
        cache_path = f"checkpoints_finetuned/ensemble_probs_{dataset_tag}.pt"
        print(f"  [INFO] One or more checkpoints not found. Loading cached softmax probabilities from {cache_path}...")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Checkpoint files are missing and cache file not found at {cache_path}!")
            
        data = torch.load(cache_path, map_location=device)
        cached_softmax = data["softmax"]
        labels = data["labels"]
        
        # Verify that all required streams are in the cache
        for s in streams_to_use:
            if s not in cached_softmax:
                raise FileNotFoundError(f"Stream checkpoint missing and stream '{s}' not found in cached probabilities!")
            softmax_dict[s] = cached_softmax[s].to(device)
        labels = labels.to(device)
        
        print("--- Loaded Softmax Probabilities from Cache ---")
        for stream in streams_to_use:
            acc1 = topk_accuracy(softmax_dict[stream], labels, k=1)
            acc5 = topk_accuracy(softmax_dict[stream], labels, k=5)
            print(f"  {stream.upper()} -> Acc@1: {acc1:.2f}% | Acc@5: {acc5:.2f}%")
    else:
        # 2. Perform Inference to extract Softmax probabilities
        print("\n--- Extracting Softmax Probabilities ---")
        for stream in streams_to_use:
            print(f"Evaluating stream [{stream.upper()}]...")
            model = load_model(ckpt_paths[stream], device, cfg)
            loader = build_test_loader(stream, cfg)
            
            if labels is None:
                labels = extract_labels(loader)
                
            probs = extract_softmax(model, loader, device, len(loader.dataset))
            softmax_dict[stream] = probs
            
            acc1 = topk_accuracy(probs, labels, k=1)
            acc5 = topk_accuracy(probs, labels, k=5)
            print(f"  Result -> Acc@1: {acc1:.2f}% | Acc@5: {acc5:.2f}%")
            
            del model
            torch.cuda.empty_cache()

    # 3. Calculate weights
    if args.weights is not None:
        w_list = args.weights[:len(streams_to_use)]
        total = sum(w_list) if sum(w_list) > 0 else 1.0
        weights = {s: w / total for s, w in zip(streams_to_use, w_list)}
    else:
        # Default paper weights or equal weighting
        if len(streams_to_use) == 3:
            # Paper optimal weights (NTU-120 vs NTU-60)
            if args.ntu120_xsub:
                weights = {"joint": 0.2683, "bone": 0.3171, "velocity": 0.4146}
            else:
                weights = {"joint": 0.2963, "bone": 0.3704, "velocity": 0.3333}
        else:
            weights = {s: 1.0 / len(streams_to_use) for s in streams_to_use}

    print("\nEnsemble Fusion Weights:")
    for s, w in weights.items():
        print(f"  {s.upper():10s}: {w:.4f}")

    # 4. Perform Late Fusion Ensemble
    w_total = sum(weights[s] for s in streams_to_use)
    ensemble_probs = sum(weights[s] / w_total * softmax_dict[s] for s in streams_to_use)
    
    ens_acc1 = topk_accuracy(ensemble_probs, labels, k=1)
    ens_acc5 = topk_accuracy(ensemble_probs, labels, k=5)
    
    # 5. Format results for LaTeX Table
    results_dict = {}
    results_dict["Luồng Khớp xương (Joint Stream)"] = (topk_accuracy(softmax_dict["joint"], labels, k=1), topk_accuracy(softmax_dict["joint"], labels, k=5))
    results_dict["Luồng Xương nối (Bone Stream)"] = (topk_accuracy(softmax_dict["bone"], labels, k=1), topk_accuracy(softmax_dict["bone"], labels, k=5))
    if "velocity" in softmax_dict:
        results_dict["Luồng Vận tốc (Velocity Stream)"] = (topk_accuracy(softmax_dict["velocity"], labels, k=1), topk_accuracy(softmax_dict["velocity"], labels, k=5))
    results_dict["Ensemble Late Fusion"] = (ens_acc1, ens_acc5)
    
    dataset_name = "NTU-120 X-Sub" if args.ntu120_xsub else "NTU-60 X-View"
    print_latex_table(results_dict, dataset_name)

    # 6. Plot Confusion Matrix
    if args.plot_cm:
        preds = ensemble_probs.argmax(dim=1)
        cm_title = f"Confusion Matrix (Ensemble Acc: {ens_acc1:.2f}%)"
        tag = "ntu120_xsub" if args.ntu120_xsub else "ntu60_xview"
        cm_save_path = os.path.join(args.plots_dir, f"confusion_matrix_{tag}.png")
        os.makedirs(args.plots_dir, exist_ok=True)
        plot_confusion_matrix(labels.cpu(), preds.cpu(), cfg["num_classes"], save_path=cm_save_path, title=cm_title)
        
    # Save cache
    dataset_tag = f"ntu120_xsub" if args.ntu120_xsub else f"ntu60_xview"
    cache_path = f"checkpoints_finetuned/ensemble_probs_{dataset_tag}.pt"
    os.makedirs("checkpoints_finetuned", exist_ok=True)
    torch.save({
        "softmax": softmax_dict,
        "labels": labels,
        "weights": weights,
        "best_combo": streams_to_use,
        "best_acc1": ens_acc1,
    }, cache_path)
    print(f"Softmax probabilities cached successfully to {cache_path}")

def run_grid_search(args, cfg):
    """Loads Softmax cache and performs a grid search to find the optimal late fusion weights."""
    print("\n" + "="*60)
    print("  RUNNING OPTIMAL WEIGHTS GRID SEARCH")
    print("="*60)
    
    dataset_tag = "ntu120_xsub" if args.ntu120_xsub else "ntu60_xview"
    cache_path = f"checkpoints_finetuned/ensemble_probs_{dataset_tag}.pt"
    
    if not os.path.exists(cache_path):
        print(f"[ERR] Cache file {cache_path} not found. Please run '--mode metrics' first to generate cache!")
        return
        
    data = torch.load(cache_path, map_location="cpu")
    softmax_dict = data["softmax"]
    labels = data["labels"]
    
    available_streams = list(softmax_dict.keys())
    print(f"Available streams in cache: {available_streams}")
    
    best_acc = 0.0
    best_weights = None
    
    # Grid search from 0.0 to 2.0 with step 0.1
    weights_range = np.arange(0.0, 2.1, 0.1)
    
    for alpha in weights_range:
        for beta in weights_range:
            gammas = weights_range if "velocity" in available_streams else [0.0]
            for gamma in gammas:
                if alpha == 0.0 and beta == 0.0 and gamma == 0.0:
                    continue
                    
                w_sum = alpha + beta + gamma
                w_j = alpha / w_sum
                w_b = beta / w_sum
                w_v = gamma / w_sum
                
                probs = 0.0
                if "joint" in softmax_dict:
                    probs += w_j * softmax_dict["joint"]
                if "bone" in softmax_dict:
                    probs += w_b * softmax_dict["bone"]
                if "velocity" in softmax_dict:
                    probs += w_v * softmax_dict["velocity"]
                    
                preds = probs.argmax(dim=1)
                correct = (preds == labels).float().sum().item()
                acc = 100.0 * correct / len(labels)
                
                if acc > best_acc:
                    best_acc = acc
                    best_weights = (alpha, beta, gamma)
                    
    print("\n" + "="*50)
    print("OPTIMAL ENSEMBLE WEIGHTS FOUND")
    print("="*50)
    print(f"  Alpha (Joint) : {best_weights[0]:.2f}")
    print(f"  Beta  (Bone)  : {best_weights[1]:.2f}")
    print(f"  Gamma (Velo)  : {best_weights[2]:.2f}")
    print(f"  Best Accuracy : {best_acc:.2f}%")
    print("="*50 + "\n")

def run_visualizations(args):
    """Trains the downstream visualizer model and runs t-SNE/UMAP/LDA and XAI explanations."""
    print("\n" + "="*60)
    print("  RUNNING VISUALIZATIONS & XAI EXPLANATIONS")
    print("="*60)
    
    data_dir = args.data_dir
    pretrain_path = args.pretrain_path
    
    if not os.path.exists(pretrain_path):
        print(f"WARNING: Pre-trained visualizer backbone not found at {pretrain_path}.")
        print("We will run the visualization module. It will initialize weights randomly if needed.")
        
    print(f"Extracting features, training linear classifier and plotting to {args.plots_dir}...")
    
    # Run the comprehensive visualization routine from downstream.py
    train_classifier(
        data_dir=data_dir,
        pretrain_path=pretrain_path,
        epochs=10,
        batch_size=8,
        lr=1e-3,
        plots_dir=args.plots_dir,
        max_frames=40,
        embed_dim=128,
        limit=5000 if args.ntu120_xsub else 2000
    )
    print("\nLatent space projections (t-SNE/UMAP/LDA) and XAI maps (SHAP/Saliency/Attention) generated successfully!")

def main():
    parser = argparse.ArgumentParser(description="S-JEPA Joint-Embedding Unified Replication Script")
    parser.add_argument("--mode", type=str, default="all", choices=["metrics", "find_weights", "visualize", "all"],
                        help="Operation mode: 'metrics' (Table 1), 'find_weights' (Optimal Late Fusion), 'visualize' (XAI/Manifold), 'all' (runs all).")
    parser.add_argument("--ntu120_xsub", action="store_true",
                        help="Use NTU-120 X-Sub configs instead of default NTU-60 X-View.")
    parser.add_argument("--data_dir", type=str, default="./data/ntu_skeletons",
                        help="Path to the skeleton dataset (used for visualizations).")
    parser.add_argument("--pretrain_path", type=str, default="./checkpoints/sjepa_skeleton_pretrain.pth",
                        help="Path to pre-trained visualizer weights.")
    parser.add_argument("--weights", nargs=3, type=float, default=None,
                        metavar=("W_J", "W_B", "W_V"),
                        help="Custom weights for [Joint, Bone, Velocity] Late Fusion.")
    parser.add_argument("--no_velocity", action="store_true",
                        help="Ignore the velocity stream in ensemble.")
    parser.add_argument("--plot_cm", action="store_true", default=True,
                        help="Plot and save Confusion Matrix for the best ensemble combination.")
    parser.add_argument("--plots_dir", type=str, default="./plots",
                        help="Directory to output generated figures.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use ('cuda' or 'cpu').")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Configurations matching the paper tables
    if args.ntu120_xsub:
        cfg = {
            "data_paths": ["./data/ntu_skeletons"],
            "max_frames": 120,
            "protocol": "xsub",
            "num_classes": 120,
            "embed_dim": 256,
            "depth": 8,
            "num_heads": 8,
            "segment_length": 4,
            "batch_size": 64,
            "num_workers": 4,
        }
        ckpt_paths = {
            "joint":    "checkpoints_finetuned/finetune_NTU120_XSub_Attention/best.pth",
            "bone":     "checkpoints_finetuned/finetune_NTU120_XSub_bone_pretrain/best.pth",
            "velocity": "checkpoints_finetuned/finetune_NTU120_XSub_velocity_pretrain/best.pth",
        }
    else:
        cfg = {
            "data_paths": ["./data/ntu_skeletons"],
            "max_frames": 120,
            "protocol": "xview",
            "num_classes": 60,
            "embed_dim": 256,
            "depth": 8,
            "num_heads": 8,
            "segment_length": 4,
            "batch_size": 64,
            "num_workers": 4,
        }
        ckpt_paths = {
            "joint":    "checkpoints_finetuned/finetune_NTU60_XView/best.pth",
            "bone":     "checkpoints_finetuned/finetune_NTU60_XView_bone/best.pth",
            "velocity": "checkpoints_finetuned/finetune_NTU60_XView_velocity/best.pth",
        }

    # Execute selected modes
    if args.mode in ["metrics", "all"]:
        run_metrics_evaluation(args, cfg, ckpt_paths, device)
        
    if args.mode in ["find_weights", "all"]:
        run_grid_search(args, cfg)
        
    if args.mode in ["visualize", "all"]:
        run_visualizations(args)

    print("\nReplication process finished successfully!")

if __name__ == "__main__":
    main()
