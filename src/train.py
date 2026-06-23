import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import read_skeleton_file, normalize_skeleton
from masking import generate_spatiotemporal_masks
from model_sjepa import SJEPA

class NTUSkeletonDataset(Dataset):
    """
    PyTorch Dataset to load NTU skeleton data.
    Pads or truncates sequences to a fixed length for batching.
    """
    def __init__(self, data_dir, max_frames=40, normalize=True, limit=None):
        self.max_frames = max_frames
        self.normalize = normalize
        self.filepaths = glob.glob(os.path.join(data_dir, "*.skeleton"))
        
        if len(self.filepaths) == 0:
            # Fallback if dataset is not yet prepared
            raise FileNotFoundError(f"No .skeleton files found in {data_dir}. Run eda.py first to create mock data!")
            
        if limit is not None and limit > 0:
            # Sort first to guarantee deterministic behavior across OS file systems
            sorted_paths = sorted(self.filepaths)
            import random
            random.seed(42)
            random.shuffle(sorted_paths)
            self.filepaths = sorted_paths[:limit]

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        # joints shape: (num_frames, 2, 25, 3)
        joints, meta = read_skeleton_file(filepath)
        
        # Normalize conditionally
        if self.normalize:
            joints = normalize_skeleton(joints)
        
        # Padding/Truncation to fixed length for batching
        num_frames = joints.shape[0]
        if num_frames >= self.max_frames:
            joints = joints[:self.max_frames]
        else:
            padding = np.zeros((self.max_frames - num_frames, 2, 25, 3), dtype=np.float32)
            joints = np.concatenate([joints, padding], axis=0)
            
        action_class = meta['action']
        
        # We convert action ID (1-120) to 0-indexed class (0-119)
        label = action_class - 1
        
        # Convert to float tensor: shape (T, 25, 3) for primary performer (body 0)
        # S-JEPA processes single performer sequences (or we can stack bodies, 
        # but single performer is standard for initial experiments)
        x = torch.from_numpy(joints[:, 0, :, :]).float()
        
        return x, label

def train_sjepa(data_dir, epochs=10, batch_size=4, lr=1e-3, checkpoint_dir="./checkpoints", plots_dir="./plots", max_frames=40, embed_dim=128, limit=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Load Dataset
    print(f"Initializing NTU Skeleton Dataset from {data_dir}...")
    dataset = NTUSkeletonDataset(data_dir, max_frames=max_frames, limit=limit)
    
    # Use multi-process data loading to speed up reading the files
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
    
    # 2. Instantiate S-JEPA Model
    model = SJEPA(temp_patch_size=4, embed_dim=embed_dim, enc_depth=3, pred_depth=2, num_heads=4)
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    print(f"Training S-JEPA on device: {device}")
    print(f"Parameters: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, EmbedDim={embed_dim}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    epoch_losses = []
    
    print("\n--- Starting S-JEPA Self-Supervised Pre-training ---")
    from tqdm import tqdm
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        # tqdm progress bar wrapped around dataloader
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
            
            loss = model.forward_pretrain(x, context_mask, target_masks)
            
            loss.backward()
            optimizer.step()
            
            model.update_target_encoder()
            
            total_loss += loss.item()
            
            # Update the progress bar postfix with current loss
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})
            
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch}/{epochs}] | Completed. Average Loss: {avg_loss:.5f}", flush=True)
        
    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "sjepa_skeleton_pretrain.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nSaved S-JEPA pre-trained weights to {checkpoint_path}")
    
    # 3. Plot and Save loss curve
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o', color='purple', lw=2)
    plt.title("S-JEPA Self-Supervised Loss Curve", fontsize=13, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "sjepa_pretrain_loss.png"), dpi=300)
    plt.close()
    print(f"Saved loss curve to {plots_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="S-JEPA Skeleton Pre-training Script")
    parser.add_argument("--data_dir", type=str, default="./data/ntu_skeletons", help="Path to skeleton dataset directory")
    parser.add_argument("--epochs", type=str, default="15", help="Number of pre-training epochs")
    parser.add_argument("--batch_size", type=str, default="8", help="Batch size")
    parser.add_argument("--lr", type=str, default="1e-3", help="Learning rate")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save weights")
    parser.add_argument("--plots_dir", type=str, default="./plots", help="Directory to save loss curves")
    parser.add_argument("--max_frames", type=str, default="40", help="Sequence max length")
    parser.add_argument("--embed_dim", type=str, default="128", help="Feature embedding dimension")
    
    parser.add_argument("--limit", type=str, default="-1", help="Limit number of samples for testing")
    
    args = parser.parse_args()
    
    # Run training
    train_sjepa(
        data_dir=args.data_dir,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        checkpoint_dir=args.checkpoint_dir,
        plots_dir=args.plots_dir,
        max_frames=int(args.max_frames),
        embed_dim=int(args.embed_dim),
        limit=int(args.limit) if int(args.limit) > 0 else None
    )

