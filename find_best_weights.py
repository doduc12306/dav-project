import os
import torch
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Grid Search Optimal Weights for S-JEPA Ensemble")
    parser.add_argument("--ntu120_xsub", action="store_true",
                        help="Dùng cấu hình NTU-120 X-Sub thay vì NTU-60 X-View mặc định")
    parser.add_argument("--suffix", type=str, default="",
                        help="Hậu tố của file cache (ví dụ: 'pretrain')")
    args = parser.parse_args()

    # Tạo dataset tag tương tự ensemble_eval.py
    dataset_tag = "ntu120_xsub" if args.ntu120_xsub else "ntu60_xview"
    if args.suffix:
        dataset_tag = f"{dataset_tag}_{args.suffix}"

    cache_path = f"checkpoints_finetuned/ensemble_probs_{dataset_tag}.pt"
    
    if not os.path.exists(cache_path):
        print(f"\n[ERR] Không tìm thấy file cache: {cache_path}")
        print("=> Vui lòng chạy lệnh ensemble_eval.py để sinh file cache trước:")
        suffix_str = f" --suffix {args.suffix}" if args.suffix else ""
        print(f"   python ensemble_eval.py --ntu120_xsub --alpha 1.0 --beta 1.0 --gamma 1.0{suffix_str}\n")
        return

    print(f"Loading Softmax cache từ {cache_path}...")
    data = torch.load(cache_path, map_location="cpu")
    softmax_dict = data["softmax"]
    labels = data["labels"]

    available_streams = list(softmax_dict.keys())
    print(f"Các luồng dữ liệu khả dụng: {available_streams}")
    
    best_acc = 0.0
    best_weights = None
    
    # Thiết lập không gian tìm kiếm từ 0.0 đến 2.0 với bước nhảy 0.1
    weights_range = np.arange(0.0, 2.1, 0.1)
    
    print(f"\nĐang quét lưới (Grid Search) tìm tổ hợp trọng số tối ưu cho tag [{dataset_tag}]...")
    
    for alpha in weights_range:
        for beta in weights_range:
            gammas = weights_range if "velocity" in available_streams else [0.0]
            for gamma in gammas:
                if alpha == 0.0 and beta == 0.0 and gamma == 0.0:
                    continue
                
                # Chuẩn hóa trọng số
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
                
                # Tính độ chính xác Acc@1
                preds = probs.argmax(dim=1)
                correct = (preds == labels).float().sum().item()
                acc = 100.0 * correct / len(labels)
                
                if acc > best_acc:
                    best_acc = acc
                    best_weights = (alpha, beta, gamma)

    print("\n" + "="*50)
    print("BỘ TRỌNG SỐ TỐI ƯU TUYỆT ĐỐI ĐÃ TÌM THẤY")
    print("="*50)
    print(f"  Alpha (Joint) : {best_weights[0]:.2f}")
    print(f"  Beta  (Bone)  : {best_weights[1]:.2f}")
    print(f"  Gamma (Velo)  : {best_weights[2]:.2f}")
    print("-"*50)
    print(f"   Accuracy Ensemble cao nhất đạt: {best_acc:.2f}%")
    print("="*50)
    
    suffix_cmd = f" --suffix {args.suffix}" if args.suffix else ""
    ntu_cmd = " --ntu120_xsub" if args.ntu120_xsub else ""
    print(f"\n Hãy chạy lệnh sau để vẽ Confusion Matrix với bộ số tối ưu này:")
    print(f"   python ensemble_eval.py{ntu_cmd} --alpha {best_weights[0]:.1f} --beta {best_weights[1]:.1f} --gamma {best_weights[2]:.1f} --plot{suffix_cmd}\n")

if __name__ == "__main__":
    main()
