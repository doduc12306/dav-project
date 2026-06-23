import os
import re
import numpy as np

# NTU RGB+D Skeleton joints connection map (used for plotting)
NTU_JOINTS_CONNECTIONS = [
    (1, 2), (2, 21), (21, 3), (3, 4),           # Head and Spine
    (21, 5), (5, 6), (6, 7), (7, 8),            # Left arm
    (21, 9), (9, 10), (10, 11), (11, 12),       # Right arm
    (1, 13), (13, 14), (14, 15), (15, 16),      # Left leg
    (1, 17), (17, 18), (18, 19), (19, 20),      # Right leg
    (22, 23), (23, 24), (24, 25)                # Fingers and thumbs (represented as extensions)
]
# Adjust 1-indexed to 0-indexed for Python arrays
NTU_CONNECTIONS = [(a - 1, b - 1) for a, b in NTU_JOINTS_CONNECTIONS]

def parse_skeleton_filename(filepath):
    """
    Parses NTU skeleton filename to extract metadata.
    Example filename: S018C001P008R001A050.skeleton
    """
    basename = os.path.basename(filepath)
    match = re.match(r'S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})', basename)
    if match:
        return {
            'setup': int(match.group(1)),
            'camera': int(match.group(2)),
            'performer': int(match.group(3)),
            'replication': int(match.group(4)),
            'action': int(match.group(5)),
            'filename': basename
        }
    return None

def read_skeleton_file(filepath):
    """
    Reads a single NTU .skeleton file and returns:
    - joints: numpy array of shape (num_frames, num_bodies, 25, 3) for X, Y, Z coordinates
    - info: dictionary with metadata
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    num_frames = int(lines[0].strip())
    idx = 1
    
    # We support up to 2 bodies (NTU 120 contains at most 2 bodies performing actions together)
    # Shape: (frames, bodies, joints, channels)
    all_frames_data = []
    
    for _ in range(num_frames):
        if idx >= len(lines):
            break
        num_bodies = int(lines[idx].strip())
        idx += 1
        
        bodies_data = []
        for b in range(num_bodies):
            if idx >= len(lines):
                break
            body_info = lines[idx].strip().split()
            # body_info has bodyID, trackingState, actionState, etc.
            idx += 1
            
            num_joints = int(lines[idx].strip())
            idx += 1
            
            joints_xyz = []
            for j in range(num_joints):
                if idx >= len(lines):
                    break
                joint_data = lines[idx].strip().split()
                # Joint data format:
                # x y z depthX depthY colorX colorY orientationW orientationX orientationY orientationZ trackingState
                x, y, z = float(joint_data[0]), float(joint_data[1]), float(joint_data[2])
                joints_xyz.append([x, y, z])
                idx += 1
            
            if len(joints_xyz) == 25:
                bodies_data.append(joints_xyz)
        
        # Keep consistent shape by padding or selecting bodies
        # Most frames have 1 or 2 bodies. We pad with zeros if < 2, or crop if > 2.
        if len(bodies_data) == 0:
            bodies_data = [[[0.0, 0.0, 0.0]] * 25] * 2
        elif len(bodies_data) == 1:
            bodies_data.append([[0.0, 0.0, 0.0]] * 25) # Pad 2nd body
        elif len(bodies_data) > 2:
            bodies_data = bodies_data[:2] # Crop to 2 bodies
            
        all_frames_data.append(bodies_data)
        
    # Shape: (num_frames, 2, 25, 3)
    joints_arr = np.array(all_frames_data, dtype=np.float32)
    metadata = parse_skeleton_filename(filepath)
    
    return joints_arr, metadata

def normalize_skeleton(joints):
    """
    Normalizes skeleton coordinates to make them translation and rotation invariant.
    Input joints shape: (frames, bodies, 25, 3)
    
    Normalization steps:
    1. Origin alignment: Translate so that the spine base (joint 0) of the main body (body 0)
       at frame 0 is at (0, 0, 0).
    2. Rotation alignment: Rotates skeletons so spine-base (joint 0) to spine-shoulder (joint 20)
       is along the Y axis, and left-shoulder (joint 4) to right-shoulder (joint 8) is along the X axis.
    """
    if joints.shape[0] == 0:
        return joints
        
    num_frames, num_bodies, num_joints, channels = joints.shape
    normalized = joints.copy()
    
    # 1. Translate origin to Spine Base (joint 0) of body 0 at frame 0
    # Spine base is index 0 (represented as joint 1 in 1-based indexing)
    origin = joints[0, 0, 0, :].copy() # shape (3,)
    normalized = normalized - origin[None, None, None, :]
    
    # 2. Rotate to align spine to Y-axis and shoulders to X-axis
    # Let's compute rotation matrix from first frame of body 0
    ref_skeleton = normalized[0, 0] # shape (25, 3)
    
    # Spine vector (Spine Shoulder [index 20] - Spine Base [index 0])
    spine_vec = ref_skeleton[20] - ref_skeleton[0]
    spine_norm = np.linalg.norm(spine_vec)
    if spine_norm > 1e-6:
        y_axis = spine_vec / spine_norm
    else:
        y_axis = np.array([0.0, 1.0, 0.0])
        
    # Shoulder vector (Right Shoulder [index 8] - Left Shoulder [index 4])
    shoulder_vec = ref_skeleton[8] - ref_skeleton[4]
    shoulder_norm = np.linalg.norm(shoulder_vec)
    if shoulder_norm > 1e-6:
        x_axis = shoulder_vec / shoulder_norm
    else:
        x_axis = np.array([1.0, 0.0, 0.0])
        
    # Project X-axis to be orthogonal to Y-axis
    x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
    x_norm = np.linalg.norm(x_axis)
    if x_norm > 1e-6:
        x_axis = x_axis / x_norm
    else:
        x_axis = np.array([1.0, 0.0, 0.0])
        
    # Z-axis is the cross product of X and Y (facing out of screen)
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Rotation matrix R
    R = np.stack([x_axis, y_axis, z_axis], axis=1) # shape (3, 3)
    
    # Apply rotation R to all coordinates across all frames and bodies
    # Shape of normalized: (F, B, 25, 3)
    # We rotate by multiplying with R
    # new_coord = R^T * coord
    normalized_reshaped = normalized.reshape(-1, 3)
    rotated_reshaped = np.dot(normalized_reshaped, R)
    normalized = rotated_reshaped.reshape(num_frames, num_bodies, num_joints, channels)
    
    return normalized

def create_mock_dataset(data_dir, num_samples=30):
    """
    Creates a folder with mock NTU skeleton files containing realistic human skeletons
    and smooth motions (e.g. sinusoidal waving/jumping) for testing visualization.
    """
    os.makedirs(data_dir, exist_ok=True)
    np.random.seed(42)
    
    # 25 joints of NTU skeleton base positions (rough coordinates mimicking human body)
    base_joints = np.array([
        [0.0, -0.2, 0.0],    # 1: Spine base
        [0.0, 0.2, 0.0],     # 2: Spine mid
        [0.0, 0.5, 0.0],     # 3: Neck
        [0.0, 0.6, 0.0],     # 4: Head
        [-0.2, 0.3, 0.0],    # 5: L shoulder
        [-0.3, 0.1, 0.0],    # 6: L elbow
        [-0.4, -0.1, 0.0],   # 7: L wrist
        [-0.45, -0.15, 0.0],  # 8: L hand
        [0.2, 0.3, 0.0],     # 9: R shoulder
        [0.3, 0.1, 0.0],     # 10: R elbow
        [0.4, -0.1, 0.0],    # 11: R wrist
        [0.45, -0.15, 0.0],   # 12: R hand
        [-0.15, -0.2, 0.0],   # 13: L hip
        [-0.15, -0.5, 0.0],   # 14: L knee
        [-0.15, -0.8, 0.0],   # 15: L ankle
        [-0.17, -0.85, 0.05], # 16: L foot
        [0.15, -0.2, 0.0],    # 17: R hip
        [0.15, -0.5, 0.0],    # 18: R knee
        [0.15, -0.8, 0.0],    # 19: R ankle
        [0.17, -0.85, 0.05],  # 20: R foot
        [0.0, 0.45, 0.0],    # 21: Spine shoulder
        [-0.46, -0.17, 0.0],  # 22: L tip
        [-0.43, -0.13, 0.0],  # 23: L thumb
        [0.46, -0.17, 0.0],   # 24: R tip
        [0.43, -0.13, 0.0]    # 25: R thumb
    ], dtype=np.float32)
    
    # NTU classes we want to mock
    actions = [23, 27, 43, 50, 72, 102] # Waving, jumping, punching, etc.
    
    for i in range(num_samples):
        # Generate valid NTU skeleton name: S{setup}C{camera}P{performer}R{rep}A{action}
        setup = np.random.randint(1, 33)
        camera = np.random.randint(1, 4)
        performer = np.random.randint(1, 107)
        rep = np.random.randint(1, 3)
        action = np.random.choice(actions)
        
        filename = f"S{setup:03d}C{camera:03d}P{performer:03d}R{rep:03d}A{action:03d}.skeleton"
        filepath = os.path.join(data_dir, filename)
        
        num_frames = np.random.randint(20, 50)
        
        # Define motion patterns based on action
        motion_amp = np.random.uniform(0.05, 0.15)
        
        with open(filepath, 'w') as f:
            f.write(f"{num_frames}\n")
            for frame in range(num_frames):
                num_bodies = 1 if action != 50 else 2 # Action 50 has 2 bodies (punching/interacting)
                f.write(f"{num_bodies}\n")
                
                for b in range(num_bodies):
                    # Mock body info line: ID, trackingState=1, others=0
                    body_id = 72057594037927936 + b
                    f.write(f"{body_id} 1 0 0 0 0 0 0 0 0\n")
                    f.write("25\n") # 25 joints
                    
                    # Compute joint positions with a temporal oscillation
                    t = frame / num_frames * 2 * np.pi
                    current_joints = base_joints.copy()
                    
                    # Add offset for 2nd body to avoid overlap
                    if b == 1:
                        current_joints[:, 0] += 0.8 # Shift right
                        
                    # Motion logic
                    if action == 23: # Waving hand
                        # Move left and right wrists/hands
                        current_joints[6:8, 0] += np.sin(t * 3) * motion_amp
                        current_joints[6:8, 1] += np.cos(t * 3) * motion_amp
                    elif action == 27: # Jumping
                        # Move entire body up and down
                        current_joints[:, 1] += np.abs(np.sin(t)) * motion_amp
                    elif action == 43: # Punching
                        # Push right hand forward along Z axis
                        current_joints[9:12, 2] += np.sin(t) * motion_amp * 2
                    
                    # Write joints
                    for j in range(25):
                        x, y, z = current_joints[j]
                        # Mock mapping coordinates for depth/color images
                        depth_x, depth_y = 250 + x*100, 200 - y*100
                        color_x, color_y = 960 + x*400, 540 - y*400
                        # Orientation and tracking state
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} {depth_x:.2f} {depth_y:.2f} {color_x:.2f} {color_y:.2f} 0 0 0 0 2\n")
    print(f"Generated {num_samples} mock skeleton files in {data_dir}")

