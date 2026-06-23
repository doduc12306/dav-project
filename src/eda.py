import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import parse_skeleton_filename, read_skeleton_file, create_mock_dataset

# Dictionary mapping NTU Action ID to human readable names for sample classes
NTU_ACTION_NAMES = {
    1: "drink water",
    2: "eat meal",
    3: "brush teeth",
    4: "brush hair",
    5: "drop",
    6: "pick up",
    7: "throw",
    8: "sit down",
    9: "stand up",
    10: "clapping",
    11: "reading",
    12: "writing",
    13: "tear up paper",
    14: "put on jacket",
    15: "take off jacket",
    16: "put on a shoe",
    17: "take off a shoe",
    18: "put on glasses",
    19: "take off glasses",
    20: "put on a hat/cap",
    21: "take off a hat/cap",
    22: "cheer up",
    23: "hand waving",
    24: "kicking something",
    25: "reach into pocket",
    26: "hopping",
    27: "jump up",
    28: "phone call",
    29: "play with phone/tablet",
    30: "type on a keyboard",
    31: "point to something",
    32: "taking a selfie",
    33: "check time (from watch)",
    34: "rub two hands",
    35: "nod head/bow",
    36: "shake head",
    37: "wipe face",
    38: "salute",
    39: "put palms together",
    40: "cross hands in front",
    41: "sneeze/cough",
    42: "staggering",
    43: "falling down",
    44: "headache",
    45: "chest pain",
    46: "back pain",
    47: "neck pain",
    48: "nausea/vomiting",
    49: "fan self",
    50: "punch/slap",
    51: "kicking",
    52: "pushing",
    53: "pat on back",
    54: "point finger",
    55: "hugging",
    56: "giving object",
    57: "touch pocket",
    58: "shaking hands",
    59: "walking towards",
    60: "walking apart",
    61: "put on headphone",
    62: "take off headphone",
    63: "shoot at basket",
    64: "bounce ball",
    65: "tennis bat swing",
    66: "juggle table tennis ball",
    67: "hush",
    68: "flick hair",
    69: "thumb up",
    70: "thumb down",
    71: "make OK sign",
    72: "make victory sign",
    73: "staple book",
    74: "counting money",
    75: "cutting nails",
    76: "cutting paper",
    77: "snap fingers",
    78: "open bottle",
    79: "sniff/smell",
    80: "squat down",
    81: "toss a coin",
    82: "fold paper",
    83: "ball up paper",
    84: "play magic cube",
    85: "apply cream on face",
    86: "apply cream on hand",
    87: "put on bag",
    88: "take off bag",
    89: "put object into bag",
    90: "take object out of bag",
    91: "open a box",
    92: "move heavy objects",
    93: "shake fist",
    94: "throw up cap/hat",
    95: "capitulate",
    96: "cross arms",
    97: "arm circles",
    98: "arm swings",
    99: "run on the spot",
    100: "butt kicks",
    101: "cross toe touch",
    102: "side kick",
    103: "yawn",
    104: "stretch oneself",
    105: "blow nose",
    106: "hit with object",
    107: "wield knife",
    108: "knock over",
    109: "grab stuff",
    110: "shoot with gun",
    111: "step on foot",
    112: "high-five",
    113: "cheers and drink",
    114: "carry object",
    115: "take a photo",
    116: "follow",
    117: "whisper",
    118: "exchange things",
    119: "support somebody",
    120: "rock-paper-scissors"
}

def analyze_dataset(data_dir):
    """
    Scans data_dir for .skeleton files, parses metadata,
    and returns a pandas DataFrame with metadata of all samples.
    """
    skeleton_files = glob.glob(os.path.join(data_dir, "*.skeleton"))
    
    if len(skeleton_files) == 0:
        print(f"Directory {data_dir} is empty! Generating mock skeleton dataset...")
        create_mock_dataset(data_dir, num_samples=50)
        skeleton_files = glob.glob(os.path.join(data_dir, "*.skeleton"))
        
    # Limit to 500 files for rapid EDA when using the full NTU dataset
    if len(skeleton_files) > 500:
        print(f"Dataset contains {len(skeleton_files)} files. Limiting EDA to 500 samples for rapid analysis.")
        sorted_files = sorted(skeleton_files)
        import random
        random.seed(42)
        random.shuffle(sorted_files)
        skeleton_files = sorted_files[:500]
        
    records = []
    print(f"Analyzing {len(skeleton_files)} skeleton files...")
    
    for filepath in skeleton_files:
        meta = parse_skeleton_filename(filepath)
        if meta is None:
            continue
            
        # Also read the skeleton to get sequence length (number of frames)
        try:
            joints, _ = read_skeleton_file(filepath)
            meta['num_frames'] = joints.shape[0]
            records.append(meta)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    df = pd.DataFrame(records)
    return df, skeleton_files

def plot_distributions(df, output_dir):
    """
    Generates and saves Seaborn/Matplotlib plots showing distributions of:
    - Action Classes
    - Camera Setup / Angles
    - Subject IDs (Performers)
    - Sequence Lengths (Frames)
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    
    # 1. Action Class Distribution (Top 15 to avoid overlaps)
    plt.figure(figsize=(9, 5))
    action_counts = df['action'].value_counts()
    
    if len(action_counts) > 15:
        action_counts_plot = action_counts.head(15)
        title = "Top 15 Most Common Action Classes"
    else:
        action_counts_plot = action_counts.sort_index()
        title = "Action Class Distribution"
        
    labels = [f"A{a}: {NTU_ACTION_NAMES.get(a, 'Unknown')}" for a in action_counts_plot.index]
    sns.barplot(x=action_counts_plot.values, y=labels, hue=labels, palette="viridis", legend=False)
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Sample Count", fontsize=11)
    plt.ylabel("Action Class", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_distribution.png"), dpi=300)
    plt.close()
    
    # 2. Sequence Length distribution (KDE / Histogram)
    plt.figure(figsize=(8, 4.5))
    sns.histplot(df['num_frames'], kde=True, color="#0f766e", bins=15)
    plt.title("Sequence Length Distribution (Frame Count)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Number of Frames", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "frame_distribution.png"), dpi=300)
    plt.close()
    
    # 3. Setup and Camera Distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Convert categories to strings to avoid seaborn warnings
    df_plot = df.copy()
    df_plot['camera_str'] = df_plot['camera'].apply(lambda x: f"Cam {x}")
    df_plot['setup_str'] = df_plot['setup'].apply(lambda x: f"Setup {x}")
    
    sns.countplot(data=df_plot, x='camera_str', ax=axes[0], hue='camera_str', palette="mako", legend=False)
    axes[0].set_title("Distribution by Camera ID", fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel("Camera ID", fontsize=10)
    axes[0].set_ylabel("Sample Count", fontsize=10)
    
    sns.countplot(data=df_plot, x='setup_str', ax=axes[1], hue='setup_str', palette="flare", legend=False)
    axes[1].set_title("Distribution by Setup ID", fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel("Setup ID", fontsize=10)
    axes[1].set_ylabel("Sample Count", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "camera_setup_distribution.png"), dpi=300)
    plt.close()
    
    print(f"Saved distribution plots to {output_dir}")

def plot_joint_dynamics(skeleton_filepath, output_dir):
    """
    Computes and plots the velocity of hands and head joints over time.
    Shows the kinetic profile of an action.
    """
    joints, meta = read_skeleton_file(skeleton_filepath)
    action_id = meta['action']
    action_name = NTU_ACTION_NAMES.get(action_id, f"Action {action_id}")
    
    frames = joints.shape[0]
    body_joints = joints[:, 0, :, :] # shape (F, 25, 3)
    
    # Calculate Euclidean displacement between frames
    diff = np.diff(body_joints, axis=0) # shape (F-1, 25, 3)
    velocities = np.linalg.norm(diff, axis=2) # shape (F-1, 25)
    
    time_steps = np.arange(1, frames)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(9, 4.5))
    plt.plot(time_steps, velocities[:, 3], label="Head (Joint 3)", color="#b91c1c", lw=2.5, linestyle="--")
    plt.plot(time_steps, velocities[:, 7], label="Left Hand (Joint 7)", color="#1d4ed8", lw=2.5)
    plt.plot(time_steps, velocities[:, 11], label="Right Hand (Joint 11)", color="#047857", lw=2.5)
    
    plt.title(f"Joint Velocity Profile: {action_name.title()}", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Frame Index", fontsize=11)
    plt.ylabel("Velocity (Coordinate Units / Frame)", fontsize=11)
    plt.legend(frameon=True, facecolor="white", edgecolor="lightgrey")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    filename = f"joint_dynamics_{action_name.replace(' ', '_').replace('/', '_')}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()
    print(f"Saved joint velocity profile as {filename} in {output_dir}")

if __name__ == "__main__":
    # Define directories
    data_directory = "./data/ntu_skeletons"
    plots_directory = "./plots"
    
    # Run analysis
    df, files = analyze_dataset(data_directory)
    
    print("\n--- Descriptive Dataset Statistics ---")
    print(df.describe())
    print("\nSample Count for each Action:")
    print(df['action'].value_counts())
    
    # Plot distributions
    plot_distributions(df, plots_directory)
    
    # Plot joint movement dynamics for a few samples
    # Select one waving sample and one jumping sample if available
    waving_samples = df[df['action'] == 23]
    jumping_samples = df[df['action'] == 27]
    
    if not waving_samples.empty:
        idx = waving_samples.index[0]
        plot_joint_dynamics(files[idx], plots_directory)
    if not jumping_samples.empty:
        idx = jumping_samples.index[0]
        plot_joint_dynamics(files[idx], plots_directory)
