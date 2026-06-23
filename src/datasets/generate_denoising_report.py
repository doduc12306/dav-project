import os
import glob
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Đảm bảo import được module từ thư mục gốc
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)
from src.datasets.ntu_dataset import read_ntu_skeleton

def check_one_file(file_path):
    try:
        if os.path.getsize(file_path) < 100: return None
        
        filename = os.path.basename(file_path)
        skel_25 = read_ntu_skeleton(file_path)
        if skel_25 is None: return None
        
        T, num_bodies_padded, J, C = skel_25.shape
        # Đếm số người thực sự tồn tại trong file (bỏ qua những người tàng hình do Zero-Padding)
        real_num_bodies = sum(1 for b in range(num_bodies_padded) if np.abs(skel_25[:, b, :, :]).sum() > 1e-6)
        
        body_results = []
        has_short = False
        has_stick = False
        motion_list = []
        
        for b in range(num_bodies_padded):
            body = skel_25[:, b, :, :]
            if np.abs(body).sum() < 1e-6: continue
            
            non_zero_mask = (np.abs(body).sum(axis=(1, 2)) > 1e-6)
            actual_len = np.sum(non_zero_mask)
            
            if actual_len <= 11:
                body_results.append(f"   -> Body {b}: Loại bỏ (Quá ngắn, {actual_len} frames)")
                has_short = True
                continue
                
            valid_frames = body[non_zero_mask]
            if len(valid_frames) == 0: continue
            
            dx = valid_frames[:, :, 0].max(axis=1) - valid_frames[:, :, 0].min(axis=1)
            dy = valid_frames[:, :, 1].max(axis=1) - valid_frames[:, :, 1].min(axis=1)
            is_valid_frame = (dx <= 0.8 * (dy + 1e-8))
            valid_ratio = np.mean(is_valid_frame)
            
            if valid_ratio < 0.3:
                body_results.append(f"   -> Body {b}: Loại bỏ (Nhiễu vật thể/bàn ghế, InvalidFrames={100*(1-valid_ratio):.1f}%)")
                has_stick = True
                continue
                
            clean_frames = valid_frames[is_valid_frame]
            if len(clean_frames) <= 1: continue
            
            motion = np.diff(clean_frames, axis=0)
            variance = np.var(motion)
            motion_list.append({"id": b, "var": variance})

        res_type = None
        if real_num_bodies >= 2 and len(motion_list) >= 2:
            res_type = "motion"
        elif has_stick:
            res_type = "stick"
        elif has_short:
            res_type = "short"


        if res_type:
            motion_list.sort(key=lambda x: x['var'], reverse=True)
            log = [f"File: {filename} (Gốc: {real_num_bodies} người)"] + body_results
            for i, m in enumerate(motion_list):
                tag = "CHỌN (Diễn viên chính)" if i < 2 else "LOẠI BỎ (Người đứng yên/Nhiễu nền)"
                log.append(f"   -> Body {m['id']}: {tag} - Mức độ chuyển động = {m['var']:.6f}")
            return {"type": res_type, "log": "\n".join(log + ["-"*50])}
    except:
        pass
    return None

def generate_report():
    print("Đang quét dữ liệu (Vui lòng đợi trong giây lát)...")
    data_paths = ["./DATA/nturgbd_skeletons_s001_to_s017/nturgb+d_skeletons", "./DATA/nturgbd_skeletons_s018_to_s032"]
    files = []
    for dp in data_paths:
        if os.path.exists(dp):
            files.extend(glob.glob(os.path.join(dp, "*.skeleton")))
    
    blacklist = set()
    if os.path.exists("DATA/missing_skeletons.txt"):
        with open("DATA/missing_skeletons.txt", 'r') as f:
            blacklist = {line.strip() for line in f if line.strip()}
    
    files = [f for f in files if os.path.basename(f).split('.')[0] not in blacklist]
    np.random.seed(42); np.random.shuffle(files)
    
    ev_short, ev_stick, ev_motion = [], [], []
    num_workers = min(os.cpu_count() or 4, 12)
    
    p1 = tqdm(total=10, desc="1. Nhiễu Quá ngắn", position=0, leave=True)
    p2 = tqdm(total=10, desc="2. Nhiễu Đồ vật", position=1, leave=True)
    p3 = tqdm(total=10, desc="3. Lọc Chuyển động", position=2, leave=True)
    
    # Quét tối đa 40.000 file để đảm bảo tìm thấy mẫu hiếm
    search_limit = 40000
    files_to_scan = files[:search_limit]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(check_one_file, f): f for f in files_to_scan}
        
        try:
            for future in as_completed(futures):
                r = future.result()
                if r:
                    if r["type"] == "motion" and len(ev_motion) < 10:
                        ev_motion.append(r["log"]); p3.update(1)
                    elif r["type"] == "stick" and len(ev_stick) < 10:
                        ev_stick.append(r["log"]); p2.update(1)
                    elif r["type"] == "short" and len(ev_short) < 10:
                        ev_short.append(r["log"]); p1.update(1)
                
                if len(ev_short) >= 10 and len(ev_stick) >= 10 and len(ev_motion) >= 10:
                    break
        except KeyboardInterrupt:
            print("\nĐã dừng theo yêu cầu của người dùng.")
            
    p1.close(); p2.close(); p3.close()
    
    # Đã sửa lại đường dẫn lưu file vào cùng thư mục với script này (src/datasets/)
    report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "denoising_evidence.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== BÁO CÁO MINH CHỨNG TIỀN XỬ LÝ (DENOISING AUDIT REPORT) ===\n\n")
        f.write("NHÓM 1: LỌC THEO ĐỘ DÀI (GHOST NOISE)\n" + "\n".join(ev_short) + "\n\n")
        f.write("NHÓM 2: LỌC THEO ĐỒ VẬT (BÀN/GHẾ SOFA)\n" + "\n".join(ev_stick) + "\n\n")
        f.write("NHÓM 3: LỌC THEO CHUYỂN ĐỘNG (MOTION VARIANCE)\n" + "\n".join(ev_motion))
    print(f"\nĐã xuất báo cáo thành công ra file: {report_path}")

if __name__ == "__main__":
    generate_report()
