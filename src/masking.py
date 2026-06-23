import numpy as np
import torch

# Define NTU joints grouped by 5 body parts for anatomically sensible masking
BODY_PARTS = {
    'torso': [0, 1, 2, 3, 20],               # Spine base, Spine mid, Neck, Head, Spine shoulder
    'left_arm': [4, 5, 6, 7, 21, 22],        # Left shoulder, elbow, wrist, hand, fingertips, thumb
    'right_arm': [8, 9, 10, 11, 23, 24],     # Right shoulder, elbow, wrist, hand, fingertips, thumb
    'left_leg': [12, 13, 14, 15],            # Left hip, knee, ankle, foot
    'right_leg': [16, 17, 18, 19]            # Right hip, knee, ankle, foot
}

BODY_PART_LIST = list(BODY_PARTS.values())

def generate_spatiotemporal_masks(num_frames, temp_patch_size, num_joints=25, 
                                 target_aspect_ratio=(0.5, 2.0), target_scale=(0.15, 0.25),
                                 context_scale=(0.6, 0.8), num_targets=3):
    """
    Generates indices for S-JEPA Context and Target tokens.
    Tokens are represented by index (t_patch_idx, joint_idx) where:
      - t_patch_idx ranges from 0 to num_frames // temp_patch_size - 1
      - joint_idx ranges from 0 to num_joints - 1
    
    Returns:
    - context_mask: Boolean tensor of shape (num_patches_t, num_joints) 
      indicating which tokens are sent to the Context Encoder.
    - target_masks: List of Boolean tensors of shape (num_patches_t, num_joints)
      each representing a target block to be predicted.
    """
    num_patches_t = num_frames // temp_patch_size
    
    # 1. Generate Target Masks (Multiple smaller spatio-temporal blocks)
    target_masks = []
    target_union = np.zeros((num_patches_t, num_joints), dtype=bool)
    
    for _ in range(num_targets):
        mask = np.zeros((num_patches_t, num_joints), dtype=bool)
        
        # Determine temporal size of target (scale of frames)
        t_scale = np.random.uniform(target_scale[0], target_scale[1])
        t_size = max(1, int(num_patches_t * t_scale))
        t_start = np.random.randint(0, num_patches_t - t_size + 1)
        t_slice = slice(t_start, t_start + t_size)
        
        # Determine spatial size of target (select 1 or 2 body parts)
        num_parts_to_mask = np.random.choice([1, 2])
        chosen_parts_idx = np.random.choice(len(BODY_PART_LIST), num_parts_to_mask, replace=False)
        
        for p_idx in chosen_parts_idx:
            joints_in_part = BODY_PART_LIST[p_idx]
            mask[t_slice, joints_in_part] = True
            
        target_masks.append(torch.from_numpy(mask))
        target_union = np.logical_or(target_union, mask)
        
    # 2. Generate Context Mask (A single large spatio-temporal block)
    # The Context Encoder should see a large part of the skeleton, 
    # but we MUST remove any overlap with targets to prevent information leak.
    context_mask = np.zeros((num_patches_t, num_joints), dtype=bool)
    
    # Choose temporal slice for Context
    c_scale = np.random.uniform(context_scale[0], context_scale[1])
    c_size = max(1, int(num_patches_t * c_scale))
    c_start = np.random.randint(0, num_patches_t - c_size + 1)
    c_slice = slice(c_start, c_start + c_size)
    
    # Context initially covers most joints in that temporal slice
    # Let's say it covers 4 out of 5 body parts
    num_parts_context = 4
    context_parts = np.random.choice(len(BODY_PART_LIST), num_parts_context, replace=False)
    for p_idx in context_parts:
        joints_in_part = BODY_PART_LIST[p_idx]
        context_mask[c_slice, joints_in_part] = True
        
    # Crucial JEPA step: Context must NOT overlap with any Target token!
    # context = context \ target_union
    context_mask = np.logical_and(context_mask, np.logical_not(target_union))
    
    # Convert target masks and context mask to torch tensors
    return torch.from_numpy(context_mask), target_masks

if __name__ == "__main__":
    # Test mask generation
    c_mask, t_masks = generate_spatiotemporal_masks(num_frames=40, temp_patch_size=4)
    print("Spatio-Temporal Grid size: (Time Patches: 10, Joints: 25)")
    print(f"Context Tokens Count: {c_mask.sum().item()} / 250")
    for idx, t_m in enumerate(t_masks):
        print(f"Target {idx+1} Tokens Count: {t_m.sum().item()} / 250")
    print("Overlap check (should be 0):", (c_mask & t_masks[0]).sum().item())
