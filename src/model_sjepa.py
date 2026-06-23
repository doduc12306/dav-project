import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletonPatchProjection(nn.Module):
    """
    Splits a skeleton sequence of shape (B, T, J, C) into spatio-temporal tokens.
    For each joint, coordinates over 'temp_patch_size' frames are flattened and
    projected into an embedding space of size 'embed_dim'.
    """
    def __init__(self, temp_patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        self.temp_patch_size = temp_patch_size
        self.proj = nn.Linear(temp_patch_size * in_channels, embed_dim)
        
    def forward(self, x):
        # x shape: (B, T, J, C)
        B, T, J, C = x.shape
        P_t = self.temp_patch_size
        N_t = T // P_t
        
        # Crop sequence to be a multiple of patch size
        x = x[:, :N_t * P_t]
        
        # Reshape to group time steps into patches
        # (B, N_t, P_t, J, C)
        x = x.view(B, N_t, P_t, J, C)
        # Permute to (B, N_t, J, P_t, C) and flatten the last dimensions
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(B, N_t, J, P_t * C)
        
        # Project each joint-time patch to embedding dimension
        # (B, N_t, J, embed_dim)
        x = self.proj(x)
        return x

class SpatioTemporalPosEmbedding(nn.Module):
    """
    Learnable spatio-temporal positional embeddings.
    Combines independent temporal embeddings (frame progression) and spatial 
    embeddings (skeleton structure layout).
    """
    def __init__(self, max_t_patches=30, num_joints=25, embed_dim=128):
        super().__init__()
        self.temp_embed = nn.Parameter(torch.zeros(max_t_patches, 1, embed_dim))
        self.joint_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        
        # Initialize with truncated normal
        nn.init.trunc_normal_(self.temp_embed, std=0.02)
        nn.init.trunc_normal_(self.joint_embed, std=0.02)
        
    def forward(self, N_t, J):
        # shape: (N_t, J, embed_dim)
        pos = self.temp_embed[:N_t] + self.joint_embed[:, :J]
        return pos

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_weights = attn.clone()
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_weights

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.attn(norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

class SkeletonViT(nn.Module):
    """
    Vision Transformer (ViT) blocks returning attention maps for visual XAI.
    """
    def __init__(self, embed_dim=128, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, return_attn=False):
        # x shape: (B, N_tokens, embed_dim)
        last_attn = None
        for block in self.blocks:
            x, last_attn = block(x)
        x = self.norm(x)
        if return_attn:
            return x, last_attn
        return x

class SJEPA(nn.Module):
    """
    Spatio-Temporal Joint-Embedding Predictive Architecture (S-JEPA) for Skeleton Data.
    """
    def __init__(self, temp_patch_size=4, embed_dim=128, 
                 enc_depth=4, pred_depth=2, num_heads=4, ema_decay=0.996):
        super().__init__()
        self.temp_patch_size = temp_patch_size
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay
        
        # 1. Patch projection and positional embedding
        self.patch_proj = SkeletonPatchProjection(temp_patch_size, 3, embed_dim)
        self.pos_embedder = SpatioTemporalPosEmbedding(max_t_patches=40, num_joints=25, embed_dim=embed_dim)
        
        # 2. Context Encoder (ViT)
        self.context_encoder = SkeletonViT(embed_dim, enc_depth, num_heads)
        
        # 3. Target Encoder (ViT) - Updated via EMA, no gradient computation
        self.target_encoder = SkeletonViT(embed_dim, enc_depth, num_heads)
        # Initialize target encoder weights same as context encoder
        self.update_target_encoder(decay=0.0)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # 4. Predictor (ViT) - Smaller transformer to predict target from context
        self.predictor = SkeletonViT(embed_dim, pred_depth, num_heads)
        
        # 5. Mask Token for prediction
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    @torch.no_grad()
    def update_target_encoder(self, decay=None):
        """
        Updates the Target Encoder weights using exponential moving average (EMA)
        of the Context Encoder.
        """
        if decay is None:
            decay = self.ema_decay
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * decay + param_q.data * (1.0 - decay)
            
    def forward_pretrain(self, x, context_mask, target_masks):
        """
        Self-supervised pre-training step.
        - x: skeleton coordinates of shape (B, T, 25, 3)
        - context_mask: Boolean tensor of shape (N_t, 25)
        - target_masks: List of Boolean tensors of shape (N_t, 25)
        """
        B, T, J, C = x.shape
        N_t = T // self.temp_patch_size
        
        # 1. Project to embeddings
        # tokens shape: (B, N_t, J, embed_dim)
        tokens = self.patch_proj(x)
        
        # 2. Add positional embedding
        pos_embed = self.pos_embedder(N_t, J) # (N_t, J, embed_dim)
        tokens = tokens + pos_embed.unsqueeze(0)
        
        # Flatten spatio-temporal coordinates to 1D sequence of tokens
        # (B, N_t * J, embed_dim)
        tokens_flat = tokens.view(B, N_t * J, self.embed_dim)
        pos_embed_flat = pos_embed.view(N_t * J, self.embed_dim)
        
        # Flatten the masks: (N_t * J)
        context_mask_flat = context_mask.view(-1)
        
        # 3. Extract Context Tokens and feed to Context Encoder
        context_tokens = tokens_flat[:, context_mask_flat] # (B, N_context, embed_dim)
        context_feats = self.context_encoder(context_tokens) # (B, N_context, embed_dim)
        
        # 4. Feed ALL tokens to Target Encoder (gradients disabled)
        with torch.no_grad():
            target_feats_all = self.target_encoder(tokens_flat) # (B, N_t * J, embed_dim)
            
        # 5. Predict each target block
        total_loss = 0.0
        
        for t_mask in target_masks:
            t_mask_flat = t_mask.view(-1)
            num_target_tokens = t_mask_flat.sum().item()
            
            if num_target_tokens == 0:
                continue
                
            # Get actual target embeddings from the Target Encoder output
            actual_targets = target_feats_all[:, t_mask_flat] # (B, N_target, embed_dim)
            
            # Prepare Predictor Inputs: Context Feats + Mask Tokens + Target Positional Embeddings
            # Gather positional embeddings for the target tokens
            target_pos = pos_embed_flat[t_mask_flat].unsqueeze(0) # (1, N_target, embed_dim)
            
            # Create mask representations for predictor
            # (B, N_target, embed_dim)
            pred_mask_tokens = self.mask_token.expand(B, num_target_tokens, -1) + target_pos
            
            # Concatenate Context and Mask Tokens
            # (B, N_context + N_target, embed_dim)
            pred_in = torch.cat([context_feats, pred_mask_tokens], dim=1)
            
            # Run Predictor
            pred_out = self.predictor(pred_in) # (B, N_context + N_target, embed_dim)
            
            # Extract predictions corresponding to the mask tokens
            predicted_targets = pred_out[:, context_feats.shape[1]:] # (B, N_target, embed_dim)
            
            # Compute MSE Loss
            loss = F.mse_loss(predicted_targets, actual_targets)
            total_loss += loss
            
        return total_loss / len(target_masks)

    def extract_features(self, x):
        """
        Extracts representations for downstream tasks (Linear Probing or Fine-tuning).
        Takes all skeleton patches and passes them through the Context Encoder.
        """
        B, T, J, C = x.shape
        N_t = T // self.temp_patch_size
        
        tokens = self.patch_proj(x)
        pos_embed = self.pos_embedder(N_t, J)
        tokens = tokens + pos_embed.unsqueeze(0)
        
        tokens_flat = tokens.view(B, N_t * J, self.embed_dim)
        
        # Pass all tokens through the trained context encoder
        feats = self.context_encoder(tokens_flat) # (B, N_t * J, embed_dim)
        
        # Global Average Pooling across spatio-temporal tokens
        global_feats = feats.mean(dim=1) # (B, embed_dim)
        return global_feats

    def extract_attention(self, x):
        """
        Extracts features and attention weights from the last layer of Context Encoder.
        x shape: (B, T, J, C)
        """
        B, T, J, C = x.shape
        N_t = T // self.temp_patch_size
        
        tokens = self.patch_proj(x)
        pos_embed = self.pos_embedder(N_t, J)
        tokens = tokens + pos_embed.unsqueeze(0)
        
        tokens_flat = tokens.view(B, N_t * J, self.embed_dim)
        
        # Pass through Context Encoder requesting attention weights
        feats, attn = self.context_encoder(tokens_flat, return_attn=True)
        return feats, attn

class ActionClassifier(nn.Module):
    """
    Classifier module for Downstream fine-tuning.
    Wraps the trained S-JEPA backbone and adds a linear projection classification layer.
    """
    def __init__(self, sjepa_backbone, num_classes=120):
        super().__init__()
        self.backbone = sjepa_backbone
        # Freeze backbone weights if doing Linear Probing
        # self.freeze_backbone()
        
        self.classifier = nn.Sequential(
            nn.Linear(sjepa_backbone.embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        # Extract features from S-JEPA backbone
        feats = self.backbone.extract_features(x) # (B, embed_dim)
        # Classify
        logits = self.classifier(feats) # (B, num_classes)
        return logits
