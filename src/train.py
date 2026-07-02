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

from src.data_utils import read_skeleton_file, normalize_skeleton
from src.masking import generate_spatiotemporal_masks
from src.model_sjepa import SJEPA

class NTUSkeletonDataset(torch.utils.data.Dataset):
    """
    Legacy PyTorch Dataset to load NTU skeleton data.
    Pads or truncates sequences to a fixed length for batching.
    Used by downstream.py for visualizations.
    """
    def __init__(self, data_dir, max_frames=40, normalize=True, limit=None):
        self.max_frames = max_frames
        self.normalize = normalize
        import glob
        self.filepaths = glob.glob(os.path.join(data_dir, "*.skeleton"))
        
        if len(self.filepaths) == 0:
            raise FileNotFoundError(f"No .skeleton files found in {data_dir}. Run eda.py first to create mock data!")
            
        if limit is not None and limit > 0:
            sorted_paths = sorted(self.filepaths)
            import random
            random.seed(42)
            random.shuffle(sorted_paths)
            self.filepaths = sorted_paths[:limit]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        joints, meta = read_skeleton_file(filepath)
        
        if self.normalize:
            joints = normalize_skeleton(joints)
        
        num_frames = joints.shape[0]
        if num_frames >= self.max_frames:
            joints = joints[:self.max_frames]
        else:
            padding = np.zeros((self.max_frames - num_frames, 2, 25, 3), dtype=np.float32)
            joints = np.concatenate([joints, padding], axis=0)
            
        action_class = meta['action']
        label = action_class - 1
        x = torch.from_numpy(joints[:, 0, :, :]).float()
        return x, label

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
    max_frames=40,
    embed_dim=128,
    enc_depth=3,
    pred_depth=2,
    num_heads=4,
    limit=None
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load Dataset
    print(f"Initializing NTU Skeleton Dataset from {data_dir}...")
    dataset = NTUSkeletonDataset(data_dir, max_frames=max_frames, limit=limit)
    print(f"Dataset initialized with {len(dataset)} files.")

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

    # 2. Instantiate S-JEPA Mock Model
    model = SJEPA(
        temp_patch_size=4,
        embed_dim=embed_dim,
        enc_depth=enc_depth,
        pred_depth=pred_depth,
        num_heads=num_heads,
        ema_decay=ema_decay
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
        
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device) # shape: (B, T, 25, 3)
            
            # Generate shared masks for this batch
            context_mask, target_masks = generate_spatiotemporal_masks(
                num_frames=max_frames, temp_patch_size=4, num_joints=25
            )
            
            context_mask = context_mask.to(device)
            target_masks = [tm.to(device) for tm in target_masks]

            optimizer.zero_grad()

            # Autocast float16 mixed precision training
            with torch.cuda.amp.autocast(enabled=use_cuda):
                loss = model.forward_pretrain(x, context_mask, target_masks)

            if use_cuda:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Smoothly update target encoder weights using EMA
            model.update_target_encoder()

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
    parser.add_argument("--max_frames", type=int, default=40, help="Sequence max length")
    parser.add_argument("--embed_dim", type=int, default=128, help="Feature embedding dimension")
    parser.add_argument("--enc_depth", type=int, default=3, help="Encoder depth")
    parser.add_argument("--pred_depth", type=int, default=2, help="Predictor depth")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of samples for testing")

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
        enc_depth=args.enc_depth,
        pred_depth=args.pred_depth,
        num_heads=args.num_heads,
        limit=args.limit if args.limit > 0 else None
    )



