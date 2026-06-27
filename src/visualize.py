import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from data_utils import read_skeleton_file, normalize_skeleton, NTU_CONNECTIONS, parse_skeleton_filename
from eda import NTU_ACTION_NAMES

def create_static_3d_skeleton(filepath, output_dir, normalize=True):
    """
    Reads an NTU skeleton file and creates a static 3D skeleton visualization
    with a sequence of 4 keyframes side-by-side to show motion.
    Saves as PNG.
    """
    joints_raw, meta = read_skeleton_file(filepath)
    filename = meta['filename']
    action_id = meta['action']
    
    # Apply normalization (translation & rotation alignment)
    if normalize:
        joints = normalize_skeleton(joints_raw)
        title_suffix = " (Normalized)"
        suffix = "_normalized"
    else:
        joints = joints_raw
        title_suffix = " (Raw Coordinates)"
        suffix = "_raw"
        
    num_frames, num_bodies, num_joints, _ = joints.shape
    
    # Select 4 keyframes across the sequence
    if num_frames >= 4:
        indices = [0, num_frames // 3, 2 * num_frames // 3, num_frames - 1]
    else:
        indices = list(range(num_frames)) + [num_frames - 1] * (4 - num_frames)
        
    fig = plt.figure(figsize=(16, 4.5))
    
    # Gather non-zero joints to calculate bounding box limits
    valid_mask = joints != 0.0
    x_valid = joints[..., 0][valid_mask[..., 0]]
    y_valid = joints[..., 1][valid_mask[..., 1]]
    z_valid = joints[..., 2][valid_mask[..., 2]]
    
    if len(x_valid) == 0:
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        z_min, z_max = -1.0, 1.0
    else:
        x_min, x_max = x_valid.min(), x_valid.max()
        y_min, y_max = y_valid.min(), y_valid.max()
        z_min, z_max = z_valid.min(), z_valid.max()
        
    # Calculate a uniform scale factor to keep aspect ratios 1:1:1
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
    mid_x = (x_max + x_min) / 2.0
    mid_y = (y_max + y_min) / 2.0
    mid_z = (z_max + z_min) / 2.0
    
    for i, f_idx in enumerate(indices):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        joints_frame = joints[f_idx]
        
        # Plot bones (lines)
        for b in range(num_bodies):
            body = joints_frame[b]
            if np.all(body == 0.0):
                continue
            for joint_a, joint_b in NTU_CONNECTIONS:
                # We swap Y and Z axes for Matplotlib's 3D projection to view it naturally (standing up)
                ax.plot(
                    [body[joint_a, 0], body[joint_b, 0]],
                    [body[joint_a, 2], body[joint_b, 2]],
                    [body[joint_a, 1], body[joint_b, 1]],
                    color='dimgrey' if b == 0 else 'darkgrey',
                    linewidth=2.5,
                    zorder=1
                )
                
        # Plot joints (scatter)
        for b in range(num_bodies):
            body = joints_frame[b]
            if np.all(body == 0.0):
                continue
            color = 'deepskyblue' if b == 0 else 'orange'
            ax.scatter(
                body[:, 0],
                body[:, 2],
                body[:, 1],
                color=color,
                edgecolors='black',
                s=35,
                alpha=0.9,
                zorder=2
            )
            
        # Uniform limits to prevent stretching
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(mid_y - max_range, mid_y + max_range)
        
        ax.set_title(f"Frame {f_idx}", fontsize=10, pad=0)
        
        # Clean up the view (remove background pane, grid lines and tick labels)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        
        # Set viewing angle
        ax.view_init(elev=15, azim=-75)
        
    action_name = NTU_ACTION_NAMES.get(action_id, f"Action A{action_id}")
    fig.suptitle(f"3D Skeleton Motion Sequence: {action_name}{title_suffix}\nFile: {filename}", fontsize=12, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{os.path.splitext(filename)[0]}{suffix}_3d.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created static 3D visualization: {output_path}")
    return output_path

if __name__ == "__main__":
    data_dir = "./data/ntu_skeletons"
    output_dir = "./plots"
    
    skeleton_files = glob.glob(os.path.join(data_dir, "*.skeleton"))
    if len(skeleton_files) == 0:
        print("Dataset directory is empty. Generating mock files first...")
        from data_utils import create_mock_dataset
        create_mock_dataset(data_dir, num_samples=10)
        skeleton_files = glob.glob(os.path.join(data_dir, "*.skeleton"))
        
    # Find a specific jumping action sample (A027) if available
    jumping_files = [f for f in skeleton_files if "A027" in f]
    selected_file = jumping_files[0] if jumping_files else skeleton_files[0]
    
    # Generate static keyframe plots for normalized coordinates
    create_static_3d_skeleton(selected_file, output_dir, normalize=True)
    # Generate static keyframe plots for raw coordinates
    create_static_3d_skeleton(selected_file, output_dir, normalize=False)
