import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.datasets.ntu_dataset import SJEPA_UnsupervisedDataset
from src.core.sjepa import sjepa_base

def train_sjepa(
    data_dir,
    epochs=15,
    batch_size=8,
    lr=1e-3,
    weight_decay=0.05,
    warmup_epochs=3,
    ema_decay=0.996,
    checkpoint_dir="./checkpoints",
    plots_dir="./plots",
    max_frames=120,
    embed_dim=256,
    depth=8,
    num_heads=8,
    predictor_depth=5,
    segment_length=4,
    limit=None,
    protocol=None
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load Dataset
    print(f"Initializing NTU Skeleton Dataset from {data_dir}...")
    dataset = SJEPA_UnsupervisedDataset(
        data_path=data_dir,
        max_frames=max_frames,
        mask_ratio=0.9,
        protocol=protocol,
        segment_length=segment_length
    )
    
    if limit is not None and limit > 0:
        # Apply sample limit for quick testing/debugging
        dataset.data_paths = dataset.data_paths[:limit]
        print(f"Sample limit applied. Using {len(dataset.data_paths)} files.")

    use_cuda = torch.cuda.is_available()
    num_workers = 4 if use_cuda else 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=use_cuda
    )

    # 2. Instantiate S-JEPA Base Model (Official Configuration)
    model = sjepa_base(
        embed_dim=embed_dim,
        num_frames=max_frames,
        depth=depth,
        num_heads=num_heads,
        predictor_depth=predictor_depth,
        segment_length=segment_length
    )
    
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    print(f"Training S-JEPA on device: {device}")
    print(f"Parameters: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, EmbedDim={embed_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. Learning Rate Scheduler with Warm-up and Cosine Annealing
    def get_lr_multiplier(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        # Cosine decay from lr to lr*0.01
        progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.01 + 0.99 * (0.5 * (1.0 + math.cos(math.pi * progress)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
    
    # 4. Mixed Precision Setup
    scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    epoch_losses = []

    print("\n--- Starting S-JEPA Self-Supervised Pre-training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]", unit="batch")
        
        for batch_idx, (x_student, x_teacher, context_idx, target_idx) in enumerate(pbar):
            x_student = x_student.to(device)
            x_teacher = x_teacher.to(device)
            context_idx = context_idx.to(device)
            target_idx = target_idx.to(device)

            optimizer.zero_grad()

            # Autocast float16 mixed precision training
            with torch.cuda.amp.autocast(enabled=use_cuda):
                loss = model(x_student, x_teacher, context_idx, target_idx)

            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Smoothly update teacher encoder weights using EMA
            model.update_teacher(m=ema_decay)

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{epochs}] | Completed. Avg Loss: {avg_loss:.5f} | LR: {current_lr:.6f}", flush=True)

    # Save pre-trained checkpoint weights
    checkpoint_path = os.path.join(checkpoint_dir, "sjepa_skeleton_pretrain.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved S-JEPA pre-trained weights to {checkpoint_path}")

    # Plot and Save loss curve
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4.5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='purple', lw=2.5)
    plt.title("S-JEPA Self-Supervised Pre-training Loss Curve", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss (MSE)", fontsize=11)
    plt.xticks(range(1, epochs + 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    loss_plot_path = os.path.join(plots_dir, "sjepa_pretrain_loss.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to {loss_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S-JEPA Skeleton Pre-training Script")
    parser.add_argument("--data_dir", type=str, default="./data/ntu_skeletons", help="Path to skeleton dataset directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of pre-training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--ema_decay", type=float, default=0.996, help="EMA decay rate")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save weights")
    parser.add_argument("--plots_dir", type=str, default="./plots", help="Directory to save loss curves")
    parser.add_argument("--max_frames", type=int, default=120, help="Sequence max length")
    parser.add_argument("--embed_dim", type=int, default=256, help="Feature embedding dimension")
    parser.add_argument("--depth", type=int, default=8, help="Transformer encoder depth")
    parser.add_argument("--num_heads", type=int, default=8, help="Attention heads")
    parser.add_argument("--predictor_depth", type=int, default=5, help="Transformer predictor depth")
    parser.add_argument("--segment_length", type=int, default=4, help="Masking segment length")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of samples for testing")
    parser.add_argument("--protocol", type=str, default=None, help="Protocol split to filter data (e.g. xsub)")

    args = parser.parse_args()

    train_sjepa(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        ema_decay=args.ema_decay,
        checkpoint_dir=args.checkpoint_dir,
        plots_dir=args.plots_dir,
        max_frames=args.max_frames,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        predictor_depth=args.predictor_depth,
        segment_length=args.segment_length,
        limit=args.limit if args.limit > 0 else None,
        protocol=args.protocol
    )


