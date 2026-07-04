"""
ensemble_eval.py — Đánh giá Ensemble Multi-Stream S-JEPA
=========================================================
Kết hợp Softmax từ 3 luồng: Joint + Bone + Velocity (Weighted Late Fusion)

CÁCH CHẠY:
    python ensemble_eval.py
    python ensemble_eval.py --weights 0.5 0.35 0.15   # Tuỳ chỉnh trọng số
    python ensemble_eval.py --no_velocity              # Chỉ Joint + Bone

CHECKPOINT MẶC ĐỊNH:
    Joint:    checkpoints_finetuned/finetune_NTU60_XView/best.pth
    Bone:     checkpoints_finetuned/finetune_NTU60_XView_bone/best.pth
    Velocity: checkpoints_finetuned/finetune_NTU60_XView_velocity/best.pth
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast

from src.core.classifier import NTUActionClassifier
from src.datasets.ntu_dataset import NTUActionDataset

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# ─────────────────────────────────────────────
#  CẤU HÌNH MẶC ĐỊNH
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "data_paths": ["./DATA/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons"],
    "max_frames": 120,
    "protocol": "xview",
    "num_classes": 60,
    "embed_dim": 256,
    "depth": 8,
    "num_heads": 8,
    "segment_length": 4,
    "batch_size": 64,
    "num_workers": 8,
}

CHECKPOINT_PATHS = {
    "joint":    "checkpoints_finetuned/finetune_NTU60_XView/best.pth",
    "bone":     "checkpoints_finetuned/finetune_NTU60_XView_bone/best.pth",
    "velocity": "checkpoints_finetuned/finetune_NTU60_XView_velocity/best.pth",
}

# ──── NTU-120 X-Sub ────
NTU120_XSUB_CONFIG = {
    "data_paths": ["./DATA/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons", "./DATA/nturgbd_skeletons_s018_to_s032"],
    "max_frames": 120,
    "protocol": "xsub",
    "num_classes": 120,
    "embed_dim": 256,
    "depth": 8,
    "num_heads": 8,
    "segment_length": 4,
    "batch_size": 64,
    "num_workers": 8,
}

NTU120_XSUB_CHECKPOINT_PATHS = {
    "joint":    "checkpoints_finetuned/finetune_NTU120_XSub_Attention/best.pth",
    "bone":     "checkpoints_finetuned/finetune_NTU120_XSub_bone_pretrain/best.pth",
    "velocity": "checkpoints_finetuned/finetune_NTU120_XSub_velocity_pretrain/best.pth",
}


# ─────────────────────────────────────────────
#  HÀM PHỤ TRỢ
# ─────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device, cfg: dict) -> NTUActionClassifier:
    """Tạo model và nạp trọng số từ checkpoint Finetune."""
    model = NTUActionClassifier(
        pretrained_path=None,           # Không cần pretrain — nạp full state_dict
        num_frames=cfg["max_frames"],
        num_classes=cfg["num_classes"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        segment_length=cfg["segment_length"],
        dropout=0.0,                    # Tắt Dropout khi eval
        drop_path=0.0,
    ).to(device)
    
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"  [OK] Loaded: {ckpt_path}")
    return model


@torch.no_grad()
def extract_softmax(model: NTUActionClassifier,
                    loader: DataLoader,
                    device: torch.device,
                    num_samples: int) -> torch.Tensor:
    """
    Trích xuất xác suất Softmax của từng mẫu trong Test Set.
    Trả về Tensor [N, num_classes] trên CPU.
    """
    all_probs = []
    for x, _ in loader:
        x = x.to(device)
        with autocast('cuda'):
            logits = model(x)
        probs = F.softmax(logits, dim=1)   # [B, C]
        all_probs.append(probs.cpu())
    return torch.cat(all_probs, dim=0)     # [N, C]


@torch.no_grad()
def extract_labels(loader: DataLoader) -> torch.Tensor:
    """Lấy toàn bộ nhãn thực từ DataLoader."""
    all_labels = []
    for _, y in loader:
        all_labels.append(y)
    return torch.cat(all_labels, dim=0)    # [N]


def topk_accuracy(probs: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Tính Top-K Accuracy từ ma trận xác suất."""
    _, topk_preds = probs.topk(k, dim=1)  # [N, k]
    correct = topk_preds.eq(labels.unsqueeze(1).expand_as(topk_preds))
    return 100.0 * correct.any(dim=1).float().mean().item()


def plot_confusion_matrix(labels, preds, num_classes, save_path="confusion_matrix.png", title="Confusion Matrix", top_k=None, bottom_k=None):
    """Vẽ và lưu ma trận nhầm lẫn."""
    NTU_120_LABELS = [
        'drink water', 'eat meal', 'brush teeth', 'brush hair', 'drop', 'pick up', 'throw', 'sit down', 'stand up', 'clapping', 
        'reading', 'writing', 'tear up paper', 'put on jacket', 'take off jacket', 'put on a shoe', 'take off a shoe', 'put on glasses', 'take off glasses', 'put on a hat/cap', 
        'take off a hat/cap', 'cheer up', 'hand waving', 'kicking something', 'reach into pocket', 'hopping', 'jump up', 'phone call', 'play with phone/tablet', 'type on a keyboard', 
        'point to something', 'taking a selfie', 'check time (from watch)', 'rub two hands', 'nod head/bow', 'shake head', 'wipe face', 'salute', 'put palms together', 'cross hands in front', 
        'sneeze/cough', 'staggering', 'falling down', 'headache', 'chest pain', 'back pain', 'neck pain', 'nausea/vomiting', 'fan self', 'punch/slap', 
        'kicking', 'pushing', 'pat on back', 'point finger', 'hugging', 'giving object', 'touch pocket', 'shaking hands', 'walking towards', 'walking apart', 
        'put on headphone', 'take off headphone', 'shoot at basket', 'bounce ball', 'tennis bat swing', 'juggle table tennis ball', 'hush', 'flick hair', 'thumb up', 'thumb down', 
        'make OK sign', 'make victory sign', 'staple book', 'counting money', 'cutting nails', 'cutting paper', 'snap fingers', 'open bottle', 'sniff/smell', 'squat down', 
        'toss a coin', 'fold paper', 'ball up paper', 'play magic cube', 'apply cream on face', 'apply cream on hand', 'put on bag', 'take off bag', 'put object into bag', 'take object out of bag', 
        'open a box', 'move heavy objects', 'shake fist', 'throw up cap/hat', 'capitulate', 'cross arms', 'arm circles', 'arm swings', 'run on the spot', 'butt kicks', 
        'cross toe touch', 'side kick', 'yawn', 'stretch oneself', 'blow nose', 'hit with object', 'wield knife', 'knock over', 'grab stuff', 'shoot with gun', 
        'step on foot', 'high-five', 'cheers and drink', 'carry object', 'take a photo', 'follow', 'whisper', 'exchange things', 'support somebody', 'rock-paper-scissors'
    ]

    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=range(num_classes))
    
    # Chuẩn hoá theo hàng (True Label) để dễ quan sát tỉ lệ nhầm lẫn
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    class_indices = np.arange(num_classes)
    
    if (top_k is not None and top_k < num_classes) or (bottom_k is not None and bottom_k < num_classes):
        # Lấy độ chính xác từng class (đường chéo của cm_norm)
        class_accs = np.diag(cm_norm)
        if bottom_k is not None:
            # Lấy bottom_k index (kết quả tệ nhất)
            selected_indices = np.argsort(class_accs)[:bottom_k]
        else:
            # Sắp xếp giảm dần và lấy top_k index
            selected_indices = np.argsort(class_accs)[::-1][:top_k]
        
        # Trích xuất ma trận con
        cm_norm = cm_norm[selected_indices][:, selected_indices]
        class_indices = selected_indices
    
    # Nếu đang chạy trên tập NTU-60 (hoặc nhỏ hơn), danh sách sẽ tự giới hạn
    class_names = [NTU_120_LABELS[i] for i in class_indices]
    
    is_subset = (top_k is not None) or (bottom_k is not None)
    plt.figure(figsize=(10, 8.5) if not is_subset else (12, 10))
    
    if not is_subset and num_classes > 30:
        # Use class IDs instead of long text names to prevent overlaps
        sns.heatmap(cm_norm, annot=False, cmap='Blues', 
                    xticklabels=[str(i+1) for i in class_indices], 
                    yticklabels=[str(i+1) for i in class_indices])
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        plt.xlabel('Predicted Class ID', fontsize=11, labelpad=10)
        plt.ylabel('True Class ID', fontsize=11, labelpad=10)
    else:
        sns.heatmap(cm_norm, annot=False if not is_subset else True, cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.xlabel('Predicted Action', fontsize=11, labelpad=10)
        plt.ylabel('True Action', fontsize=11, labelpad=10)
    
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [ĐÃ LƯU] Ma trận nhầm lẫn -> {save_path}")


def build_test_loader(modality: str, cfg: dict) -> DataLoader:
    """Tạo DataLoader cho tập Test với modality chỉ định."""
    ds = NTUActionDataset(
        data_path=cfg["data_paths"],
        max_frames=cfg["max_frames"],
        split='test',
        protocol=cfg["protocol"],
        modality=modality,
    )
    return DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,                  # QUAN TRỌNG: giữ thứ tự để align nhãn
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if cfg["num_workers"] > 0 else None,
    )


# ─────────────────────────────────────────────
#  HÀM CHÍNH
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="S-JEPA Multi-Stream Ensemble Evaluation")
    parser.add_argument("--alpha", type=float, default=None, help="Trọng số cho Joint stream (α)")
    parser.add_argument("--beta", type=float, default=None, help="Trọng số cho Bone stream (β)")
    parser.add_argument("--gamma", type=float, default=None, help="Trọng số cho Velocity stream (γ)")
    parser.add_argument("--weights", nargs=3, type=float, default=None,
                        metavar=("W_JOINT", "W_BONE", "W_VEL"),
                        help="Trọng số cho [Joint, Bone, Velocity].")
    parser.add_argument("--no_velocity", action="store_true",
                        help="Chỉ dùng Joint + Bone (bỏ qua Velocity)")
    parser.add_argument("--load_cache", action="store_true",
                        help="Nạp kết quả Softmax từ cache (ensemble_probs.pt) để tinh chỉnh trọng số nhanh")
    parser.add_argument("--plot", action="store_true",
                        help="Vẽ ma trận nhầm lẫn (Confusion Matrix) cho kết quả tốt nhất")
    parser.add_argument("--plot_topk", type=int, default=None,
                        help="Chỉ vẽ Confusion Matrix cho Top-K class có kết quả tốt nhất (VD: 20)")
    parser.add_argument("--plot_bottomk", type=int, default=None,
                        help="Chỉ vẽ Confusion Matrix cho Bottom-K class có kết quả tệ nhất (VD: 20)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda hoặc cpu")
    parser.add_argument("--ntu120_xsub", action="store_true",
                        help="Dùng cấu hình NTU-120 X-Sub thay vì NTU-60 X-View mặc định")
    parser.add_argument("--suffix", type=str, default="",
                        help="Hậu tố thêm vào tên file ảnh và cache (ví dụ: 'pretrain')")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.ntu120_xsub:
        cfg = NTU120_XSUB_CONFIG
        ckpt_paths = NTU120_XSUB_CHECKPOINT_PATHS
        print("[Dataset] NTU-120 X-Sub")
    else:
        cfg = DEFAULT_CONFIG
        ckpt_paths = CHECKPOINT_PATHS
        print("[Dataset] NTU-60 X-View")

    print("\n" + "="*60)
    print("  S-JEPA MULTI-STREAM ENSEMBLE EVALUATION")
    print("="*60)
    print(f"  Device: {device}")
    print(f"  Protocol: {cfg['protocol'].upper()}")
    print(f"  Num Classes: {cfg['num_classes']}")
    print("="*60 + "\n")

    # ── Tự động xử lý Hậu tố (Suffix) cho checkpoints ──
    suffix_paths = {}
    for s, path in ckpt_paths.items():
        if s == "joint":
            suffix_paths[s] = path  # Luồng Joint thường giữ nguyên hoặc lấy checkpoint tốt nhất
            continue
        
        # Lấy thư mục cha và tên file
        dir_path, filename = os.path.split(path)
        
        # Bóc tách mọi hậu tố có thể có để tìm tên cơ sở (baseline)
        base_dir = dir_path
        for suffix_to_strip in ["_pretrain", "_from_pretrain"]:
            if base_dir.endswith(suffix_to_strip):
                base_dir = base_dir[:-len(suffix_to_strip)]
        
        # Các ứng viên có thể có trên đĩa
        candidates = [
            dir_path,                           # Giữ nguyên đường dẫn cấu hình
            base_dir,                           # Nhóm mặc định không hậu tố
            f"{base_dir}_from_pretrain",        # Nhóm pretrain (dạng _from_pretrain)
            f"{base_dir}_pretrain",             # Nhóm pretrain (dạng _pretrain)
        ]
        if args.suffix:
            candidates.insert(0, f"{base_dir}_{args.suffix}")
            candidates.insert(1, f"{base_dir}_from_{args.suffix}")
            
        resolved_path = None
        for cand in candidates:
            cand_path = os.path.join(cand, filename)
            if os.path.exists(cand_path):
                resolved_path = cand_path
                break
                
        if resolved_path:
            suffix_paths[s] = resolved_path
        else:
            suffix_paths[s] = path  # Fallback
            
    ckpt_paths = suffix_paths

    # ── Kiểm tra checkpoint tồn tại ──
    streams_to_use = ["joint", "bone"]
    if not args.no_velocity:
        if args.load_cache:
            streams_to_use.append("velocity")
        else:
            vel_path = ckpt_paths["velocity"]
            if os.path.exists(vel_path):
                streams_to_use.append("velocity")
            else:
                print(f"  [WARN] Velocity checkpoint không tìm thấy -> bỏ qua Velocity")
                print(f"    ({vel_path})\n")

    if not args.load_cache:
        for s in streams_to_use:
            if not os.path.exists(ckpt_paths[s]):
                raise FileNotFoundError(f"Checkpoint không tìm thấy: {ckpt_paths[s]}")

    # ── Bước 1: Load Softmax (từ Cache hoặc Inference) ──
    softmax_dict = {}
    labels = None
    dataset_tag = f"ntu120_xsub" if args.ntu120_xsub else f"ntu60_xview"
    if args.suffix:
        dataset_tag = f"{dataset_tag}_{args.suffix}"
    cache_path = f"checkpoints_finetuned/ensemble_probs_{dataset_tag}.pt"

    if args.load_cache and os.path.exists(cache_path):
        print(f"[Bước 1/3] Nạp Softmax từ Cache: {cache_path}")
        cache_data = torch.load(cache_path, map_location="cpu")
        cached_softmax = cache_data["softmax"]
        labels = cache_data["labels"]
        
        # Chỉ lấy các stream được yêu cầu và có trong cache
        for s in streams_to_use:
            if s in cached_softmax:
                softmax_dict[s] = cached_softmax[s]
                acc1 = topk_accuracy(softmax_dict[s], labels, k=1)
                print(f"  [OK] {s.upper():8s} (từ cache) | Acc@1: {acc1:.2f}%")
            else:
                print(f"  [ERR] {s.upper():8s} không có trong cache! Vui lòng chạy lại không có --load_cache.")
                return

    else:
        print("[Bước 1/3] Nạp Models & Trích xuất Softmax...")
        for stream in streams_to_use:
            print(f"\n  Luồng [{stream.upper()}]:")
            model = load_model(ckpt_paths[stream], device, cfg)
            loader = build_test_loader(stream, cfg)

            if labels is None:
                labels = extract_labels(loader)   # Chỉ cần lấy 1 lần (nhãn giống nhau)

            probs = extract_softmax(model, loader, device, len(loader.dataset))
            softmax_dict[stream] = probs          # [N, C] CPU

            acc1 = topk_accuracy(probs, labels, k=1)
            acc5 = topk_accuracy(probs, labels, k=5)
            print(f"  → Single-stream Acc@1: {acc1:.2f}%  |  Acc@5: {acc5:.2f}%")

            del model
            torch.cuda.empty_cache()

    # ── Bước 2: Thiết lập trọng số Ensemble ──
    print(f"\n[Bước 2/3] Thiết lập trọng số Ensemble...")
    
    if any(v is not None for v in [args.alpha, args.beta, args.gamma]):
        # Ưu tiên alpha, beta, gamma (theo công thức P_ensemble = a*P_j + b*P_b + g*P_v)
        raw_w = {
            "joint":    args.alpha if args.alpha is not None else 0.0,
            "bone":     args.beta  if args.beta  is not None else 0.0,
            "velocity": args.gamma if args.gamma is not None else 0.0,
        }
        # Chỉ giữ lại các stream đang dùng và chuẩn hoá
        subset_w = [raw_w[s] for s in streams_to_use]
        total_w = sum(subset_w) if sum(subset_w) > 0 else 1.0
        weights = {s: raw_w[s] / total_w for s in streams_to_use}
        
        print(f"  Dùng trọng số tùy chỉnh:")
        print(f"    Công thức: P_ens = {raw_w['joint']}*P_joint + {raw_w['bone']}*P_bone + {raw_w['velocity']}*P_vel")
    
    elif args.weights is not None:
        # Trọng số do người dùng chỉ định qua list --weights
        w_list = args.weights[:len(streams_to_use)]
        total = sum(w_list) if sum(w_list) > 0 else 1.0
        weights = {s: w / total for s, w in zip(streams_to_use, w_list)}
        print("  Dùng trọng số tuỳ chỉnh (từ --weights):")
    else:
        # Auto weight: tỉ lệ theo độ chính xác đơn luồng
        single_accs = {s: topk_accuracy(softmax_dict[s], labels, k=1)
                       for s in streams_to_use}
        total_acc = sum(single_accs.values())
        weights = {s: acc / total_acc for s, acc in single_accs.items()}
        print("  Tự động tính trọng số theo độ chính xác đơn luồng:")

    for s, w in weights.items():
        print(f"    {s.upper():10s}: {w:.4f}")

    # ── Bước 3: Weighted Ensemble ──
    print(f"\n[Bước 3/3] Tổng hợp Ensemble...")

    # Thử tất cả các tổ hợp: đơn → cặp đôi → full ensemble
    stream_list = streams_to_use

    # Đơn luồng
    combos = [[s] for s in stream_list]
    # Cặp đôi
    for i in range(len(stream_list)):
        for j in range(i + 1, len(stream_list)):
            combos.append([stream_list[i], stream_list[j]])
    # Full ensemble (chỉ thêm nếu có >= 3 luồng)
    if len(stream_list) >= 3:
        combos.append(stream_list)

    # ── In bảng kết quả ──
    SEPARATOR = "─" * 54
    print("\n" + SEPARATOR)
    print(f"  {'Combo':<30} {'Acc@1':>8}  {'Acc@5':>8}")
    print(SEPARATOR)

    best_acc1  = 0.0
    best_combo = None
    best_probs = None

    for combo in combos:
        w_total        = sum(weights[s] for s in combo)
        ensemble_probs = sum(weights[s] / w_total * softmax_dict[s] for s in combo)

        acc1 = topk_accuracy(ensemble_probs, labels, k=1)
        acc5 = topk_accuracy(ensemble_probs, labels, k=5)

        # Nhãn hiển thị: ký tự đầu viết hoa (Joint→J, Bone→B, Velocity→V)
        tag = " + ".join(s[0].upper() for s in combo)

        is_best = ">>" if acc1 > best_acc1 else "  "
        print(f"  {is_best} {tag:<28} {acc1:>7.2f}%  {acc5:>7.2f}%")

        if acc1 > best_acc1:
            best_acc1  = acc1
            best_combo = combo
            best_probs = ensemble_probs

    print(SEPARATOR)
    print(f"\nKẾT QUẢ TỐT NHẤT:")
    best_tag = " + ".join(s[0].upper() for s in best_combo)
    print(f"   Combo:  {best_tag}")
    print(f"   Acc@1:  {best_acc1:.2f}%")
    print(f"   Acc@5:  {topk_accuracy(best_probs, labels, k=5):.2f}%")
    print("=" * 60 + "\n", flush=True)

    # Lưu softmax để dùng lại sau
    torch.save({
        "softmax": softmax_dict,
        "labels": labels,
        "weights": weights,
        "best_combo": best_combo,
        "best_acc1": best_acc1,
    }, cache_path)
    print(f"  [ĐÃ LƯU] Softmax cache -> {cache_path}", flush=True)

    # ── Bước 4: Vẽ Ma trận nhầm lẫn (nếu yêu cầu) ──
    if args.plot or args.plot_topk is not None or args.plot_bottomk is not None:
        print(f"\n[Bước 4/4] Vẽ Ma trận nhầm lẫn cho kết quả tốt nhất...")
        # Lấy class dự đoán từ ensemble_probs tốt nhất
        preds = best_probs.argmax(dim=1)
        cm_title = f"Confusion Matrix (Ensemble Acc: {best_acc1:.2f}%)"
        
        if args.plot_bottomk:
            cm_title = f"Bottom {args.plot_bottomk} Worst Classes - " + cm_title
            cm_save_path = f"confusion_matrix_worst_{dataset_tag}.png"
        elif args.plot_topk:
            cm_title = f"Top {args.plot_topk} Best Classes - " + cm_title
            cm_save_path = f"confusion_matrix_best_{dataset_tag}.png"
        else:
            cm_save_path = f"confusion_matrix_{dataset_tag}.png"
            
        plot_confusion_matrix(labels, preds, cfg["num_classes"], save_path=cm_save_path, title=cm_title, top_k=args.plot_topk, bottom_k=args.plot_bottomk)
        print(f"  [ĐÃ LƯU] Ảnh Ma trận nhầm lẫn -> {cm_save_path}")


if __name__ == "__main__":
    main()
