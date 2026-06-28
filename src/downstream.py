import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import shap

from train import NTUSkeletonDataset
from model_sjepa import SJEPA, ActionClassifier
from eda import NTU_ACTION_NAMES
from data_utils import parse_skeleton_filename

def train_classifier(data_dir, pretrain_path, epochs=15, batch_size=8, lr=1e-3, plots_dir="./plots", max_frames=40, embed_dim=128, limit=None):
    os.makedirs(plots_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Dataset
    print(f"Loading datasets for downstream classification from {data_dir}...")
    dataset = NTUSkeletonDataset(data_dir, max_frames=max_frames, limit=limit)
    
    # 2. Instantiate S-JEPA and Load Pretrained weights
    sjepa_backbone = SJEPA(temp_patch_size=4, embed_dim=embed_dim, enc_depth=3, pred_depth=2, num_heads=4)
    if os.path.exists(pretrain_path):
        print(f"Loading S-JEPA weights from {pretrain_path}...")
        sjepa_backbone.load_state_dict(torch.load(pretrain_path, map_location=device))
    else:
        print("WARNING: Pre-trained weights not found. Initializing backbone randomly...")
        
    # 3. Extract Features Offline (Normalized and Raw)
    sjepa_backbone.eval()
    sjepa_backbone = sjepa_backbone.to(device)
    
    use_cuda = torch.cuda.is_available()
    num_workers = 4 if use_cuda else 0
    
    # Loader for normalized features extraction (large batch size)
    extract_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
    
    features_norm_list = []
    labels_class_list = []
    labels_camera_list = []
    
    print("\n--- Extracting S-JEPA Normalized Features Offline ---")
    from tqdm import tqdm
    with torch.no_grad():
        for idx_batch, (x, y) in enumerate(tqdm(extract_loader, desc="Normalized Extraction", unit="batch")):
            x = x.to(device)
            feats = sjepa_backbone.extract_features(x)
            features_norm_list.append(feats.cpu())
            labels_class_list.append(y)
            
            # Retrieve camera metadata from file paths
            start_idx = idx_batch * 256
            for offset in range(x.size(0)):
                filepath = dataset.filepaths[start_idx + offset]
                meta = parse_skeleton_filename(filepath)
                labels_camera_list.append(meta['camera'])
                
    features_norm = torch.cat(features_norm_list, dim=0)
    labels_class = torch.cat(labels_class_list, dim=0)
    labels_camera = np.array(labels_camera_list)
    
    # Loader for raw (unnormalized) features extraction
    import copy
    raw_dataset = copy.deepcopy(dataset)
    raw_dataset.normalize = False
    raw_extract_loader = DataLoader(raw_dataset, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
    
    features_raw_list = []
    print("\n--- Extracting S-JEPA Raw Features Offline ---")
    with torch.no_grad():
        for x, _ in tqdm(raw_extract_loader, desc="Raw Extraction", unit="batch"):
            x = x.to(device)
            feats = sjepa_backbone.extract_features(x)
            features_raw_list.append(feats.cpu())
            
    features_raw = torch.cat(features_raw_list, dim=0)
    
    # 4. Split Pre-extracted Features into Train (80%) and Test (20%)
    train_size = int(0.8 * len(dataset))
    
    g = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=g)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    train_features = features_norm[train_idx]
    train_labels = labels_class[train_idx]
    test_features = features_norm[test_idx]
    test_labels = labels_class[test_idx]
    
    # Create TensorDataset for fast training
    from torch.utils.data import TensorDataset
    train_dataset_feats = TensorDataset(train_features, train_labels)
    test_dataset_feats = TensorDataset(test_features, test_labels)
    
    train_loader = DataLoader(train_dataset_feats, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset_feats, batch_size=batch_size, shuffle=False)
    
    # Create downstream classifier (Action Recognition for 120 classes)
    model = ActionClassifier(sjepa_backbone, num_classes=120)
    model = model.to(device)
    model.freeze_backbone()
    
    print("S-JEPA Backbone frozen. Running linear probing evaluation...")
    print(f"Parameters: Epochs={epochs}, BatchSize={batch_size}, LR={lr}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=0.01)
    
    train_losses = []
    test_accs = []
    
    print("\n--- Starting Downstream Linear Probing Classifier Training (Fast MLP-only) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}]", unit="batch")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model.classifier(x)  # Direct MLP forward pass!
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})
            
        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model.classifier(x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        test_acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        test_accs.append(test_acc)
        print(f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_loss:.5f} | Test Accuracy: {test_acc * 100:.2f}%", flush=True)
        
    # Plot training metrics
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=(8, 4))
    
    color = '#dc2626'
    ax1.set_xlabel('Khung Hình/Epoch', fontsize=11)
    ax1.set_ylabel('Độ Hao Hụt Bộ Phân Lớp (Classifier Loss)', color=color, fontsize=11)
    ax1.plot(range(1, epochs+1), train_losses, color=color, marker='s', lw=2, label="Loss")
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = '#2563eb'
    ax2.set_ylabel('Độ Chính Xác Tập Test (Test Accuracy %)', color=color, fontsize=11)
    ax2.plot(range(1, epochs+1), [acc * 100 for acc in test_accs], color=color, marker='o', lw=2, label="Accuracy")
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Đường Cong Huấn Luyện Bộ Phân Lớp Downstream", fontsize=13, fontweight='bold', pad=15)
    fig.tight_layout()
    plt.savefig(os.path.join(plots_dir, "downstream_training_curves.png"), dpi=300)
    plt.close()
    
    # 3. Final Evaluation & Confusion Matrix
    print("\n--- Final Model Evaluation ---")
    model.eval()
    with torch.no_grad():
        test_logits = model.classifier(test_features.to(device))
        all_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
        all_targets = test_labels.numpy()
        
    # Print metrics report
    unique_classes = np.unique(all_targets)
    class_names = [NTU_ACTION_NAMES.get(c + 1, f"A{c+1}") for c in unique_classes]
    
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, labels=unique_classes, target_names=class_names))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=unique_classes)
    plt.figure(figsize=(10, 8.5))
    if len(unique_classes) > 20:
        # Use abstract heatmap without tick labels for large number of classes
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
        plt.xlabel("Predicted Class", fontsize=11, labelpad=10)
        plt.ylabel("True Class", fontsize=11, labelpad=10)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.xlabel("Predicted Class", fontsize=11, labelpad=10)
        plt.ylabel("True Class", fontsize=11, labelpad=10)
        
    plt.title("Downstream Classifier Confusion Matrix", fontsize=13, fontweight='bold', pad=15)
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix plot to {plots_dir}")
    
    # 4. Latent Space Visualization using t-SNE, UMAP, and LDA
    visualize_latent_space(features_norm, features_raw, labels_class, labels_camera, plots_dir, dataset=dataset)
    
    # 5. Model Attention Visualization (Explainable AI - Attention Rollout)
    visualize_attention(sjepa_backbone, dataset, plots_dir, device)
    
    # 6. Gradient-based Saliency Visualization (Explainable AI - Gradient Saliency)
    visualize_saliency(model, dataset, plots_dir, device)
    
    # 7. SHAP Explanation (Explainable AI - Shapley Additive Explanations)
    visualize_shap(model, features_norm, labels_class, plots_dir, device)
            


def visualize_latent_space(features_norm, features_raw, labels_class, labels_camera, plots_dir, dataset=None):
    """
    Projects pre-extracted S-JEPA embeddings to 2D via t-SNE, UMAP, and LDA, and visualizes.
    """
    if isinstance(features_norm, torch.Tensor):
        features_norm = features_norm.numpy()
    if isinstance(features_raw, torch.Tensor):
        features_raw = features_raw.numpy()
    if isinstance(labels_class, torch.Tensor):
        labels_class = labels_class.numpy()
    if isinstance(labels_camera, torch.Tensor):
        labels_camera = labels_camera.numpy()
        
    # Run t-SNE
    perplexity = min(30, len(features_norm) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    
    print("Projecting normalized features using t-SNE...")
    features_norm_tsne = tsne.fit_transform(features_norm)
    
    print("Projecting raw features using t-SNE...")
    features_raw_tsne = tsne.fit_transform(features_raw)
    
    # Run UMAP
    print("Projecting normalized features using UMAP...")
    n_samples = len(features_norm)
    reducer = umap.UMAP(n_neighbors=min(15, n_samples - 1), min_dist=0.1, random_state=42)
    features_norm_umap = reducer.fit_transform(features_norm)
    
    # Run LDA
    print("Projecting normalized features using LDA...")
    unique_classes = np.unique(labels_class)
    lda = LinearDiscriminantAnalysis(n_components=min(2, len(unique_classes) - 1))
    features_norm_lda = lda.fit_transform(features_norm, labels_class)
    if features_norm_lda.shape[1] == 1:
        features_norm_lda = np.hstack([features_norm_lda, np.zeros_like(features_norm_lda)])
        
    # Filter classes to top 7 for clearer visualization
    if len(unique_classes) > 7:
        counts = np.bincount(labels_class)
        top_classes = np.argsort(counts)[-7:]
        target_classes = np.intersect1d(unique_classes, top_classes)
    else:
        target_classes = unique_classes
        
    # Plot 1: Normalized Latent Space colored by Action Class (t-SNE only)
    plt.figure(figsize=(10, 6))
    colors_class = sns.color_palette("Set1", len(target_classes))
    for idx, label in enumerate(target_classes):
        mask = labels_class == label
        class_name = NTU_ACTION_NAMES.get(label + 1, f"Action A{label+1}")
        plt.scatter(features_norm_tsne[mask, 0], features_norm_tsne[mask, 1], 
                    color=colors_class[idx], label=class_name, alpha=0.8, edgecolors='k', s=45, zorder=2)
    plt.title("t-SNE Projection of Normalized Representations\n(Colored by Action Class)", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, borderaxespad=0.)
    plt.grid(True, linestyle="--", alpha=0.5, zorder=1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tsne_latent_space_by_class.png"), dpi=300)
    plt.close()
    
    # Plot 2: Normalized Latent Space colored by Camera ID
    plt.figure(figsize=(8, 6))
    unique_cams = np.unique(labels_camera)
    colors_cam = sns.color_palette("Set1", len(unique_cams))
    for idx, cam in enumerate(unique_cams):
        mask = labels_camera == cam
        plt.scatter(features_norm_tsne[mask, 0], features_norm_tsne[mask, 1], 
                    color=colors_cam[idx], label=f"Camera C{cam:03d}", alpha=0.8, edgecolors='k', s=60, zorder=2)
    plt.title("t-SNE of Normalized Representations (Colored by Camera ID)\n[View-Invariance Test]", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='lightgrey', loc='upper right')
    plt.grid(True, linestyle="--", alpha=0.5, zorder=1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tsne_latent_space_by_camera.png"), dpi=300)
    plt.close()
    
    # Plot 3: Raw Latent Space colored by Camera ID
    plt.figure(figsize=(8, 6))
    for idx, cam in enumerate(unique_cams):
        mask = labels_camera == cam
        plt.scatter(features_raw_tsne[mask, 0], features_raw_tsne[mask, 1], 
                    color=colors_cam[idx], label=f"Camera C{cam:03d}", alpha=0.8, edgecolors='k', s=60, zorder=2)
    plt.title("t-SNE of Raw (Unnormalized) Representations (Colored by Camera ID)\n[Exposing Camera Bias]", fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=10)
    plt.ylabel("t-SNE Dimension 2", fontsize=10)
    plt.legend(frameon=True, facecolor='white', edgecolor='lightgrey', loc='upper right')
    plt.grid(True, linestyle="--", alpha=0.5, zorder=1)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "tsne_latent_space_raw_by_camera.png"), dpi=300)
    plt.close()
 
    # Plot 4: t-SNE vs UMAP vs LDA comparison grid
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    
    # Subplot A: t-SNE
    for idx, label in enumerate(target_classes):
        mask = labels_class == label
        axes[0].scatter(features_norm_tsne[mask, 0], features_norm_tsne[mask, 1], 
                        color=colors_class[idx], alpha=0.8, edgecolors='k', s=35, zorder=2)
    axes[0].set_title("t-SNE Projection", fontsize=11, fontweight='bold', pad=8)
    axes[0].set_xlabel("t-SNE Dim 1", fontsize=9)
    axes[0].set_ylabel("t-SNE Dim 2", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5, zorder=1)
    
    # Subplot B: UMAP
    for idx, label in enumerate(target_classes):
        mask = labels_class == label
        axes[1].scatter(features_norm_umap[mask, 0], features_norm_umap[mask, 1], 
                        color=colors_class[idx], alpha=0.8, edgecolors='k', s=35, zorder=2)
    axes[1].set_title("UMAP Projection", fontsize=11, fontweight='bold', pad=8)
    axes[1].set_xlabel("UMAP Dim 1", fontsize=9)
    axes[1].set_ylabel("UMAP Dim 2", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5, zorder=1)
    
    # Subplot C: LDA
    for idx, label in enumerate(target_classes):
        mask = labels_class == label
        axes[2].scatter(features_norm_lda[mask, 0], features_norm_lda[mask, 1], 
                        color=colors_class[idx], alpha=0.8, edgecolors='k', s=35, zorder=2)
    axes[2].set_title("LDA Projection (Supervised)", fontsize=11, fontweight='bold', pad=8)
    axes[2].set_xlabel("LDA Dim 1", fontsize=9)
    axes[2].set_ylabel("LDA Dim 2", fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5, zorder=1)
    
    fig.suptitle("Comparison of Dimensionality Reduction Methods on S-JEPA Representations", fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "latent_comparison_tsne_umap_lda.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE, UMAP, and LDA comparative analysis plots in {plots_dir}")

    # Export projection coordinates as a CSV file for interactive Streamlit dashboard
    import pandas as pd
    if dataset is not None:
        df_proj = pd.DataFrame({
            "file_name": [os.path.basename(p) for p in dataset.filepaths[:len(features_norm)]],
            "tsne_norm_x": features_norm_tsne[:, 0],
            "tsne_norm_y": features_norm_tsne[:, 1],
            "tsne_raw_x": features_raw_tsne[:, 0],
            "tsne_raw_y": features_raw_tsne[:, 1],
            "umap_norm_x": features_norm_umap[:, 0],
            "umap_norm_y": features_norm_umap[:, 1],
            "lda_norm_x": features_norm_lda[:, 0],
            "lda_norm_y": features_norm_lda[:, 1],
            "class_id": labels_class + 1,
            "class_name": [NTU_ACTION_NAMES.get(l + 1, f"Action A{l+1}") for l in labels_class],
            "camera_id": labels_camera
        })
        csv_path = os.path.join(plots_dir, "latent_projections.csv")
        df_proj.to_csv(csv_path, index=False)
        print(f"Saved interactive projections to {csv_path}")

def visualize_saliency(model, dataset, plots_dir, device):
    """
    Computes gradient-based saliency maps for Waving Hand and Jumping actions.
    Saliency is calculated as the gradient of the class score with respect to
    the input joint coordinates.
    """
    print("\nComputing gradient-based saliency maps (XAI - Gradient Saliency)...")
    model.eval()
    
    waving_idx = None
    jumping_idx = None
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        action_class = label + 1
        if action_class == 23 and waving_idx is None:
            waving_idx = idx
        elif action_class == 27 and jumping_idx is None:
            jumping_idx = idx
        if waving_idx is not None and jumping_idx is not None:
            break
            
    samples = []
    if waving_idx is not None:
        samples.append((waving_idx, "Waving hand", 22))
    if jumping_idx is not None:
        samples.append((jumping_idx, "Jumping", 26))
        
    for idx, action_name, class_idx in samples:
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        x.requires_grad = True
        
        logits = model(x)
        score = logits[0, class_idx]
        
        model.zero_grad()
        score.backward()
        
        saliency = x.grad.abs().squeeze(0).cpu().numpy()
        saliency_2d = np.linalg.norm(saliency, axis=2)
        
        if saliency_2d.max() > 0:
            saliency_2d = saliency_2d / saliency_2d.max()
            
        plt.figure(figsize=(10, 6))
        body_parts_names = {
            3: "Head", 7: "L Hand", 11: "R Hand", 15: "L Foot", 19: "R Foot", 0: "Spine Base", 20: "Spine Shoulder"
        }
        joint_labels = [f"J{j} ({body_parts_names[j]})" if j in body_parts_names else f"J{j}" for j in range(25)]
        
        sns.heatmap(saliency_2d.T, cmap="Oranges", xticklabels=5, yticklabels=joint_labels)
        plt.title(f"Spatio-Temporal Gradient Saliency Heatmap\nAction: {action_name}", fontsize=13, fontweight='bold')
        plt.xlabel("Frame Index")
        plt.ylabel("Joint")
        plt.tight_layout()
        
        filename_heatmap = f"saliency_heatmap_{action_name.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(plots_dir, filename_heatmap), dpi=300)
        plt.close()
        
        avg_saliency = saliency_2d.mean(axis=0)
        if avg_saliency.sum() > 0:
            avg_saliency = avg_saliency / avg_saliency.sum()
            
        plt.figure(figsize=(10, 4))
        plt.bar(range(25), avg_saliency, color="darkorange", edgecolor='k', alpha=0.9)
        plt.xticks(range(25), joint_labels, rotation=90, fontsize=8)
        plt.title(f"Average Joint Saliency Profile\nAction: {action_name}", fontsize=12, fontweight='bold')
        plt.ylabel("Saliency Share")
        plt.grid(axis='y', linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        filename_bar = f"saliency_joints_{action_name.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(plots_dir, filename_bar), dpi=300)
        plt.close()
        print(f"Saved saliency plots for {action_name}")

def visualize_shap(model, features_norm, labels_class, plots_dir, device):
    """
    Computes SHAP values using pre-extracted S-JEPA latent features.
    """
    print("\nComputing SHAP values for latent feature explanation (XAI - SHAP)...")
    model.eval()
    
    if isinstance(features_norm, torch.Tensor):
        features_norm = features_norm.numpy()
    if isinstance(labels_class, torch.Tensor):
        labels_class = labels_class.numpy()
        
    # Downsample for faster computation
    np.random.seed(42)
    bg_size = min(100, len(features_norm))
    bg_idx = np.random.choice(len(features_norm), bg_size, replace=False)
    background = features_norm[bg_idx]
    
    explain_size = min(100, len(features_norm))
    explain_idx = np.random.choice(len(features_norm), explain_size, replace=False)
    features_to_explain = features_norm[explain_idx]
    
    def predict_logits(feats_np):
        feats_tensor = torch.tensor(feats_np, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits_tensor = model.classifier(feats_tensor)
        return logits_tensor.cpu().numpy()
        
    masker = shap.maskers.Independent(background)
    explainer = shap.Explainer(predict_logits, masker)
    
    shap_values = explainer(features_to_explain)
    
    features = features_to_explain
    labels = labels_class[explain_idx]
    
    unique_labels = np.unique(labels)
    class_to_explain_waving = 22 if 22 in unique_labels else (unique_labels[0] if len(unique_labels) > 0 else 0)
    class_to_explain_jumping = 26 if 26 in unique_labels else (unique_labels[min(1, len(unique_labels)-1)] if len(unique_labels) > 0 else 0)
    
    class_name_waving = NTU_ACTION_NAMES.get(class_to_explain_waving + 1, f"Action A{class_to_explain_waving+1}")
    class_name_jumping = NTU_ACTION_NAMES.get(class_to_explain_jumping + 1, f"Action A{class_to_explain_jumping+1}")
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[:, :, class_to_explain_waving], features, show=False, plot_type="bar")
    plt.title(f"SHAP Feature Importance for Latent Representations\nAction: {class_name_waving}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary_waving.png"), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[:, :, class_to_explain_jumping], features, show=False, plot_type="bar")
    plt.title(f"SHAP Feature Importance for Latent Representations\nAction: {class_name_jumping}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary_jumping.png"), dpi=300)
    plt.close()
    
    print(f"Saved SHAP summary plots in {plots_dir}")

def visualize_attention(sjepa, dataset, plots_dir, device):
    """
    Extracts self-attention weights from Context Encoder, calculates joint-level
    attention scores, and plots a bar chart showing what the model focuses on
    for different actions.
    """
    print("\nExtracting self-attention maps for model explanation (XAI)...")
    sjepa.eval()
    
    # Find sample indices for Waving hand (23) and Jumping (27)
    waving_idx = None
    jumping_idx = None
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        action_class = label + 1
        if action_class == 23 and waving_idx is None:
            waving_idx = idx
        elif action_class == 27 and jumping_idx is None:
            jumping_idx = idx
            
        if waving_idx is not None and jumping_idx is not None:
            break
            
    samples = []
    if waving_idx is not None:
        samples.append((waving_idx, "Waving hand"))
    if jumping_idx is not None:
        samples.append((jumping_idx, "Jumping"))
        
    for idx, action_name in samples:
        x, _ = dataset[idx] # (T, 25, 3)
        x = x.unsqueeze(0).to(device)
        
        # Extract features and attention map
        # feats shape: (1, N_tokens, embed_dim)
        # attn shape: (1, num_heads, N_tokens, N_tokens) where N_tokens = N_t * 25
        feats, attn = sjepa.extract_attention(x)
        
        # Average over batch and heads: shape (N_tokens, N_tokens)
        attn = attn.mean(dim=1).squeeze(0).cpu().numpy()
        
        num_tokens = attn.shape[0]
        N_t = num_tokens // 25
        
        # Reshape to (N_t, 25, N_t, 25)
        attn_reshaped = attn.reshape(N_t, 25, N_t, 25)
        
        # Sum attention directed to each key joint across all query tokens
        joint_scores = attn_reshaped.mean(axis=(0, 1, 2)) # shape (25,)
        if joint_scores.sum() > 0:
            joint_scores = joint_scores / joint_scores.sum() # Normalize
        
        # Group 25 joints into 5 body parts for display
        body_part_mapping = {
            "Torso": [0, 1, 2, 3, 20],
            "Left Arm": [4, 5, 6, 7, 21, 22],
            "Right Arm": [8, 9, 10, 11, 23, 24],
            "Left Leg": [12, 13, 14, 15],
            "Right Leg": [16, 17, 18, 19]
        }
        
        part_scores = {}
        for part_name, joints_list in body_part_mapping.items():
            part_scores[part_name] = joint_scores[joints_list].sum()
            
        # Plot attention scores grouped by body parts
        plt.figure(figsize=(6, 4))
        parts = list(part_scores.keys())
        scores = list(part_scores.values())
        
        colors = ['#f59e0b', '#3b82f6', '#10b981', '#ef4444', '#8b5cf6']
        plt.bar(parts, scores, color=colors, edgecolor='k', alpha=0.9)
        plt.title(f"S-JEPA Body Part Attention Map\nAction: {action_name}", fontsize=12, fontweight='bold')
        plt.ylabel("Attention Share")
        plt.ylim(0, 0.7)
        plt.grid(axis='y', linestyle="--", alpha=0.6)
        plt.tight_layout()
        
        filename = f"attention_{action_name.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(plots_dir, filename), dpi=300)
        plt.close()
        print(f"Saved attention map plot for {action_name} as {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Downstream Action Classifier & Visualization")
    parser.add_argument("--data_dir", type=str, default="./data/ntu_skeletons", help="Path to skeleton dataset directory")
    parser.add_argument("--pretrain_path", type=str, default="./checkpoints/sjepa_skeleton_pretrain.pth", help="Path to pre-trained weights")
    parser.add_argument("--epochs", type=str, default="10", help="Number of epochs for downstream training")
    parser.add_argument("--batch_size", type=str, default="8", help="Batch size")
    parser.add_argument("--lr", type=str, default="1e-3", help="Learning rate")
    parser.add_argument("--plots_dir", type=str, default="./plots", help="Directory to save output plots")
    parser.add_argument("--max_frames", type=str, default="40", help="Sequence max length")
    parser.add_argument("--embed_dim", type=str, default="128", help="Feature embedding dimension")
    
    parser.add_argument("--limit", type=str, default="-1", help="Limit number of samples for testing")
    
    args = parser.parse_args()
    
    train_classifier(
        data_dir=args.data_dir,
        pretrain_path=args.pretrain_path,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        plots_dir=args.plots_dir,
        max_frames=int(args.max_frames),
        embed_dim=int(args.embed_dim),
        limit=int(args.limit) if int(args.limit) > 0 else None
    )

