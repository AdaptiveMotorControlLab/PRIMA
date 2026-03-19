import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
import pickle as pkl
from ...utils.geometry import rot6d_to_rotmat, aa_to_rotmat
from ..components.pose_transformer import TransformerDecoder


def build_smal_head(cfg):
    smal_head_type = cfg.MODEL.SMAL_HEAD.get('TYPE', 'amr')

    if smal_head_type == 'new_bio_pose_transformer_decoder':
        return NewBioGuidedSMALPoseDecoder(cfg)
    else:
        raise ValueError('Unknown SMAL head type: {}'.format(smal_head_type))







class NewBioGuidedSMALPoseDecoder(nn.Module):
    """
    Bio-Guided SMAL Decoder with Pose Token Aggregation

    Final version:
    - Query tokens = [param token] + [2D keypoint tokens (optional)] + [3D keypoint tokens (optional)]
    - SAM3D-body-style layer-wise keypoint token updates:
        * 2D: predict (x,y) in [-0.5,0.5] from kp2d tokens -> token_augment position encoding
              + grid_sample image features at predicted locations -> add into kp2d token embeddings
              + invalid_mask (out-of-bounds and optional vis mask) zeroes updates
        * 3D: predict (x,y,z) from kp3d tokens -> pelvis-normalize -> token_augment position encoding
    - token_augment is injected by feeding (token_embeddings + token_augment) into each decoder layer.
    - Only param token (index 0) is used to regress pose/betas/cam deltas.
    - Outputs:
        pred_smal_params: dict with global_orient/pose/betas and optional keypoints_2d/3d
        pred_cam: [B,3]
        extra_outputs: includes bio-guided shape_feat/init_betas and pred_smal_params_list
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ========== Basic config ==========
        self.joint_rep_type = cfg.MODEL.SMAL_HEAD.get("JOINT_REP", "6d")
        self.joint_rep_dim = {"6d": 6, "aa": 3}[self.joint_rep_type]
        self.npose = self.joint_rep_dim * (cfg.SMAL.NUM_JOINTS + 1)

        # ========== Dimensions ==========
        self.decoder_dim = cfg.MODEL.SMAL_HEAD.get("DECODER_DIM", 1024)
        context_dim = cfg.MODEL.SMAL_HEAD.get("IN_CHANNELS", 1024)
        num_layers = cfg.MODEL.SMAL_HEAD.get("NUM_DECODER_LAYERS", 4)
        num_heads = cfg.MODEL.SMAL_HEAD.get("NUM_HEADS", 8)
        mlp_ratio = cfg.MODEL.SMAL_HEAD.get("MLP_RATIO", 4.0)

        # keypoint config
        self.use_keypoint_2d_tokens = cfg.MODEL.SMAL_HEAD.get("USE_KEYPOINT_2D_TOKENS", False)
        self.use_keypoint_3d_tokens = cfg.MODEL.SMAL_HEAD.get("USE_KEYPOINT_3D_TOKENS", False)
        self.num_keypoints = cfg.SMAL.get("NUM_KEYPOINTS", 26)
        self.keypoint_token_update = cfg.MODEL.SMAL_HEAD.get("KEYPOINT_TOKEN_UPDATE", False)

        # 2D update: whether to inject sampled image feature into kp2d tokens
        self.kp2d_inject_image_feat = cfg.MODEL.SMAL_HEAD.get("KP2D_INJECT_IMAGE_FEAT", True)

        # IEF iters
        self.ief_iters = cfg.MODEL.SMAL_HEAD.get("IEF_ITERS", 3)

        # pelvis indices
        self.pelvis_idx = cfg.SMAL.get("PELVIS_IDX", [0, 1])

        # ========== Test-time optimization config ==========
        self._tta_mode = False  # Track if in test-time adaptation mode

        # ========== [Coarse] Bio prior ==========
        self.bio_to_betas_init = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 41),
        )
        self.shape_projector = nn.Sequential(
            nn.Linear(41, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

        # ========== Init pose/cam ==========
        self.init_pose = nn.Parameter(torch.zeros(1, self.npose))
        self.init_cam = nn.Parameter(torch.tensor([[0.9, 0, 0]], dtype=torch.float32))

        # params -> param token
        param_dim = self.npose + 41 + 3
        self.param_to_token = nn.Sequential(
            nn.Linear(param_dim, self.decoder_dim),
            nn.LayerNorm(self.decoder_dim),
            nn.ReLU(),
        )

        # ========== Keypoint token embeddings ==========
        if self.use_keypoint_2d_tokens:
            self.keypoint_2d_embeddings = nn.Embedding(self.num_keypoints, self.decoder_dim)
            nn.init.normal_(self.keypoint_2d_embeddings.weight, std=0.02)

            # (x,y) -> token augment
            self.keypoint_2d_pos_encoder = nn.Sequential(
                nn.Linear(2, 256),
                nn.ReLU(),
                nn.Linear(256, self.decoder_dim),
            )
            # sampled image feat -> token dim (add into token embeddings)
            self.keypoint_2d_feat_linear = nn.Linear(self.decoder_dim, self.decoder_dim)

        if self.use_keypoint_3d_tokens:
            self.keypoint_3d_embeddings = nn.Embedding(self.num_keypoints, self.decoder_dim)
            nn.init.normal_(self.keypoint_3d_embeddings.weight, std=0.02)

            # (x,y,z) -> token augment
            self.keypoint_3d_pos_encoder = nn.Sequential(
                nn.Linear(3, 256),
                nn.ReLU(),
                nn.Linear(256, self.decoder_dim),
            )

        # ========== Per-token intermediate heads (predict from kp tokens themselves) ==========
        if self.keypoint_token_update:
            if self.use_keypoint_2d_tokens:
                self.kp2d_from_tokens = nn.Sequential(
                    nn.Linear(self.decoder_dim, self.decoder_dim),
                    nn.ReLU(),
                    nn.Linear(self.decoder_dim, 2),
                )
            if self.use_keypoint_3d_tokens:
                self.kp3d_from_tokens = nn.Sequential(
                    nn.Linear(self.decoder_dim, self.decoder_dim),
                    nn.ReLU(),
                    nn.Linear(self.decoder_dim, 3),
                )

        # ========== Image feature projection + pos encoding ==========
        self.image_proj = nn.Identity() if context_dim == self.decoder_dim else nn.Linear(context_dim, self.decoder_dim)
        self.image_pos_encoding = PositionalEncoding2D(self.decoder_dim)

        # ========== Transformer decoder layers ==========
        self.layers = nn.ModuleList(
            [
                PoseTransformerDecoderLayer(
                    d_model=self.decoder_dim,
                    nhead=num_heads,
                    dim_feedforward=int(self.decoder_dim * mlp_ratio),
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.decoder_dim)

        # ========== Regression heads (param token only) ==========
        self.decpose = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_dim, self.npose),
        )
        self.decshape = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_dim, 41),
        )
        self.deccam = nn.Sequential(
            nn.Linear(self.decoder_dim, self.decoder_dim // 2),
            nn.ReLU(),
            nn.Linear(self.decoder_dim // 2, 3),
        )

    # --------------------------
    # helpers: query token build
    # --------------------------
    def _build_query_tokens(self, pred_pose, pred_betas, pred_cam):
        B = pred_pose.shape[0]
        tokens = []

        params = torch.cat([pred_pose, pred_betas, pred_cam], dim=1)  # [B, param_dim]
        param_token = self.param_to_token(params).unsqueeze(1)  # [B,1,D]
        tokens.append(param_token)

        kp2d_start = None
        kp3d_start = None

        if self.use_keypoint_2d_tokens:
            kp2d_start = sum(t.shape[1] for t in tokens)
            kp2d_tokens = self.keypoint_2d_embeddings.weight.unsqueeze(0).expand(B, -1, -1).contiguous()
            tokens.append(kp2d_tokens)

        if self.use_keypoint_3d_tokens:
            kp3d_start = sum(t.shape[1] for t in tokens)
            kp3d_tokens = self.keypoint_3d_embeddings.weight.unsqueeze(0).expand(B, -1, -1).contiguous()
            tokens.append(kp3d_tokens)

        token_embeddings = torch.cat(tokens, dim=1)  # [B,Nq,D]
        token_augment = torch.zeros_like(token_embeddings)

        return token_embeddings, token_augment, kp2d_start, kp3d_start

    # --------------------------
    # helpers: updates
    # --------------------------
    def _kp2d_update(self, token_embeddings, token_augment, image_features, kp2d_start, H, W, vis_mask=None):
        """
        SAM3D-body-style 2D keypoint token update.

        image_features: [B, HW, D] projected + pos-encoded, with HW=H*W (expected 12*16)
        vis_mask: optional [B,N] bool (True=valid)
        """
        if not (self.keypoint_token_update and self.use_keypoint_2d_tokens):
            return token_embeddings, token_augment, None

        B = token_embeddings.shape[0]
        N = self.num_keypoints

        kp_tokens = token_embeddings[:, kp2d_start : kp2d_start + N, :]  # [B,N,D]

        # predict coords in [-0.5,0.5]
        pred_xy = self.kp2d_from_tokens(kp_tokens)
        pred_xy = torch.tanh(pred_xy) * 0.5

        # invalid mask (out of bounds + optional vis)
        pred_xy_01 = pred_xy + 0.5
        invalid = (
            (pred_xy_01[..., 0] < 0.0)
            | (pred_xy_01[..., 0] > 1.0)
            | (pred_xy_01[..., 1] < 0.0)
            | (pred_xy_01[..., 1] > 1.0)
        )
        if vis_mask is not None:
            invalid = invalid | (~vis_mask)
        valid = (~invalid).unsqueeze(-1).float()  # [B,N,1]

        # update token_augment slice
        token_augment = token_augment.clone()
        token_augment[:, kp2d_start : kp2d_start + N, :] = self.keypoint_2d_pos_encoder(pred_xy) * valid

        # inject sampled image feature into kp2d tokens (optional)
        if self.kp2d_inject_image_feat:
            img = image_features.view(B, H, W, self.decoder_dim).permute(0, 3, 1, 2).contiguous()  # [B,D,H,W]
            grid = (pred_xy * 2.0).unsqueeze(2)  # [B,N,1,2] in [-1,1]

            sampled = (
                F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
                .squeeze(3)
                .permute(0, 2, 1)
                .contiguous()
            )  # [B,N,D]

            sampled = sampled * valid
            token_embeddings = token_embeddings.clone()
            token_embeddings[:, kp2d_start : kp2d_start + N, :] += self.keypoint_2d_feat_linear(sampled)

        return token_embeddings, token_augment, pred_xy

    def _kp3d_update(self, token_embeddings, token_augment, kp3d_start):
        if not (self.keypoint_token_update and self.use_keypoint_3d_tokens):
            return token_embeddings, token_augment, None

        N = self.num_keypoints
        kp_tokens = token_embeddings[:, kp3d_start : kp3d_start + N, :]  # [B,N,D]

        pred_xyz = self.kp3d_from_tokens(kp_tokens)  # [B,N,3]

        # pelvis normalize
        pelvis_center = pred_xyz[:, self.pelvis_idx, :].mean(dim=1, keepdim=True)  # [B,1,3]
        pred_xyz_norm = pred_xyz - pelvis_center

        token_augment = token_augment.clone()
        token_augment[:, kp3d_start : kp3d_start + N, :] = self.keypoint_3d_pos_encoder(pred_xyz_norm)

        return token_embeddings, token_augment, pred_xyz

    # --------------------------
    # forward
    # --------------------------
    def forward(self, x, keypoint_coords_2d=None, keypoint_coords_3d=None, **kwargs):
        """
        Inputs:
            x: [B, Hp*Wp+1, C] image tokens from backbone concatenated with bio token 
            (BioCLIP token is the last token in the sequence)

        Note:
            keypoint_coords_2d can optionally provide a vis/conf mask: [B,N,3] (x,y,vis)
            We do NOT inject GT coords into tokens by default; they are used only as optional masking.
        """
        B = x.shape[0]

        # ---- 数据预处理 ----
        # 处理 4D 输入 (B, C, H, W)
        if len(x.shape) == 4:
            x = einops.rearrange(x, 'b c h w -> b (h w) c')

        bio_token = x[:, -1, :]  # [B, C] - BioCLIP token 是最后一个 token
        image_features = x[:, :-1, :]  # [B, H*W, C] - 剩余的图像特征

        # ---- Coarse bio shape ----
        init_betas = self.bio_to_betas_init(bio_token)  # [B,41]
        shape_feat = F.normalize(self.shape_projector(init_betas), dim=1)

        # ---- Image features 投影 ----
        # 只对图像特征进行投影，不包括 bio token
        image_features = self.image_proj(image_features)  # [B,HW,D]

        # Your backbone: vit crop 256x192 with patch16 => Hp=12, Wp=16
        H, W = 12, 16
        assert image_features.shape[1] == H * W, f"Expected HW={H*W}, got {image_features.shape[1]}"

        img_pos = self.image_pos_encoding(H, W).to(image_features.device)  # [HW,D]
        image_features = image_features + img_pos.unsqueeze(0)

        # ---- init params ----
        pred_pose = self.init_pose.expand(B, -1)
        pred_betas = init_betas
        pred_cam = self.init_cam.expand(B, -1)

        pred_pose_list, pred_betas_list, pred_cam_list = [], [], []
        pred_keypoints_2d_list, pred_keypoints_3d_list = [], []

        # Optional visibility mask from provided 2D keypoints
        vis_mask = None
        if keypoint_coords_2d is not None and keypoint_coords_2d.shape[-1] == 3:
            vis_mask = keypoint_coords_2d[..., 2] > 0  # [B,N]

        # ---- IEF loop ----
        for _ in range(self.ief_iters):
            token_embeddings, token_augment, kp2d_start, kp3d_start = self._build_query_tokens(
                pred_pose, pred_betas, pred_cam
            )

            # ---- Transformer layers ----
            for layer_idx, layer in enumerate(self.layers):
                # inject dynamic augment
                tokens_in = token_embeddings + token_augment
                token_embeddings = layer(tokens_in, image_features)

                # layer-wise token update (skip last layer)
                if self.keypoint_token_update and (layer_idx < len(self.layers) - 1):
                    if self.use_keypoint_2d_tokens:
                        token_embeddings, token_augment, pred_xy = self._kp2d_update(
                            token_embeddings, token_augment, image_features, kp2d_start, H, W, vis_mask=vis_mask
                        )
                        if pred_xy is not None:
                            pred_keypoints_2d_list.append(pred_xy)

                    if self.use_keypoint_3d_tokens:
                        token_embeddings, token_augment, pred_xyz = self._kp3d_update(
                            token_embeddings, token_augment, kp3d_start
                        )
                        if pred_xyz is not None:
                            pred_keypoints_3d_list.append(pred_xyz)

            # ---- Regress deltas from param token ----
            token_embeddings = self.norm(token_embeddings)
            param_token_out = token_embeddings[:, 0, :]

            delta_pose = self.decpose(param_token_out)
            delta_betas = self.decshape(param_token_out)
            delta_cam = self.deccam(param_token_out)

            pred_pose = pred_pose + delta_pose
            pred_betas = pred_betas + delta_betas
            pred_cam = pred_cam + delta_cam

            pred_pose_list.append(pred_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        # ---- Convert joint representation ----
        joint_conversion_fn = {
            "6d": rot6d_to_rotmat,
            "aa": lambda y: aa_to_rotmat(y.view(-1, 3).contiguous()),
        }[self.joint_rep_type]

        pred_smal_params_list = {
            "pose": torch.cat(
                [joint_conversion_fn(p).view(B, -1, 3, 3)[:, 1:, :, :] for p in pred_pose_list],
                dim=0,
            ),
            "betas": torch.cat(pred_betas_list, dim=0),
            "cam": torch.cat(pred_cam_list, dim=0),
            "keypoints_2d": torch.cat(pred_keypoints_2d_list, dim=0) if len(pred_keypoints_2d_list) else None,
            "keypoints_3d": torch.cat(pred_keypoints_3d_list, dim=0) if len(pred_keypoints_3d_list) else None,
        }

        pred_pose_mat = joint_conversion_fn(pred_pose).view(B, self.cfg.SMAL.NUM_JOINTS + 1, 3, 3)
        pred_smal_params = {
            "global_orient": pred_pose_mat[:, [0]],
            "pose": pred_pose_mat[:, 1:],
            "betas": pred_betas,
        }

        # expose final predicted keypoints for losses
        if self.keypoint_token_update:
            if self.use_keypoint_2d_tokens and len(pred_keypoints_2d_list):
                pred_smal_params["keypoints_2d"] = pred_keypoints_2d_list[-1]
            if self.use_keypoint_3d_tokens and len(pred_keypoints_3d_list):
                pred_smal_params["keypoints_3d"] = pred_keypoints_3d_list[-1]

        extra_outputs = {
            "shape_feat": shape_feat,
            "init_betas": init_betas,
            "pred_smal_params_list": pred_smal_params_list,
        }
        return pred_smal_params, pred_cam, extra_outputs

    # --------------------------
    # Test-time optimization helpers
    # --------------------------
    def freeze_all_except_keypoint_tokens(self):
        """
        Freeze all parameters except keypoint token embeddings and their prediction heads.
        Use this before test-time optimization.
        """
        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze only keypoint-related parameters
        if self.use_keypoint_2d_tokens:
            for param in self.keypoint_2d_embeddings.parameters():
                param.requires_grad = True
            for param in self.keypoint_2d_pos_encoder.parameters():
                param.requires_grad = True
            for param in self.keypoint_2d_feat_linear.parameters():
                param.requires_grad = True
            if self.keypoint_token_update:
                for param in self.kp2d_from_tokens.parameters():
                    param.requires_grad = True
        
        if self.use_keypoint_3d_tokens:
            for param in self.keypoint_3d_embeddings.parameters():
                param.requires_grad = True
            for param in self.keypoint_3d_pos_encoder.parameters():
                param.requires_grad = True
            if self.keypoint_token_update:
                for param in self.kp3d_from_tokens.parameters():
                    param.requires_grad = True
        
        self._tta_mode = True
        print("[TTA] Frozen all parameters except keypoint tokens")
    
    def freeze_backbone_only(self):
        """
        Freeze only backbone, keep SMAL head trainable.
        Use for full SMAL parameter + keypoint optimization.
        """
        # Unfreeze all SMAL head parameters
        for param in self.parameters():
            param.requires_grad = True
        
        self._tta_mode = True
        print("[TTA] SMAL head fully trainable (backbone frozen separately)")
    
    def freeze_except_regression_heads(self):
        """
        Freeze everything except the final regression heads (pose/shape/cam) and keypoint embeddings.
        Keep transformer frozen to preserve pretrained representations.
        """
        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False
        
        # Unfreeze only the final regression heads (small MLPs)
        for param in self.decpose.parameters():
            param.requires_grad = True
        for param in self.decshape.parameters():
            param.requires_grad = True
        for param in self.deccam.parameters():
            param.requires_grad = True
        
        # Unfreeze ONLY keypoint embeddings (learned tokens, NOT position encoders)
        if self.use_keypoint_2d_tokens:
            self.keypoint_2d_embeddings.weight.requires_grad = True
        
        if self.use_keypoint_3d_tokens:
            self.keypoint_3d_embeddings.weight.requires_grad = True
        
        # DO NOT unfreeze transformer - keep pretrained representations
        # DO NOT unfreeze param_to_token - keep initial token mapping stable
        
        self._tta_mode = True
        print("[TTA] Frozen all except regression heads and keypoint embeddings")
    
    def unfreeze_all(self):
        """Restore all parameters to trainable state."""
        for param in self.parameters():
            param.requires_grad = True
        self._tta_mode = False
        print("[TTA] Unfrozen all parameters")
    
    def get_tta_parameters(self, mode='keypoints_only'):
        """
        Get list of parameters that should be optimized during test-time adaptation.
        MUST match what's unfrozen by freeze methods!
        
        Args:
            mode: 'keypoints_only', 'regression_heads', or 'all'
        """
        params = []
        
        # Keypoint embeddings only (NOT position encoders or feature linears)
        if mode in ['keypoints_only', 'regression_heads', 'all']:
            if self.use_keypoint_2d_tokens:
                params.append(self.keypoint_2d_embeddings.weight)
            
            if self.use_keypoint_3d_tokens:
                params.append(self.keypoint_3d_embeddings.weight)
        
        # Regression heads only (NO transformer or param_to_token)
        if mode in ['regression_heads', 'all']:
            params.extend(list(self.decpose.parameters()))
            params.extend(list(self.decshape.parameters()))
            params.extend(list(self.deccam.parameters()))
        
        return params




class PoseTransformerDecoderLayer(nn.Module):
    """
    单层 Transformer Decoder for Pose Token Aggregation
    包含:  Self-Attention (tokens 交互) + Cross-Attention (tokens ← image) + FFN
    """
    
    def __init__(self, d_model=1024, nhead=8, dim_feedforward=4096, dropout=0.1):
        super().__init__()
        
        # Self-Attention:  tokens 之间交互
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Cross-Attention: tokens 从图像聚合信息
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, tokens, image_features):
        """
        Args:
            tokens:  [B, N_tokens, C] - 所有 tokens (pose + keypoints)
            image_features:  [B, N_pixels, C] - 图像特征
        
        Returns: 
            tokens:  [B, N_tokens, C] - 更新后的 tokens
        """
        
        # Self-Attention:  pose token 从 keypoint tokens 聚合信息
        attn_output, _ = self.self_attn(tokens, tokens, tokens)
        tokens = tokens + self.dropout1(attn_output)
        tokens = self.norm1(tokens)
        
        # Cross-Attention: tokens 从图像特征聚合视觉信息
        attn_output, _ = self.cross_attn(
            query=tokens,
            key=image_features,
            value=image_features,
        )
        tokens = tokens + self.dropout2(attn_output)
        tokens = self.norm2(tokens)
        
        # Feed-Forward Network
        ffn_output = self.ffn(tokens)
        tokens = tokens + ffn_output
        tokens = self.norm3(tokens)
        
        return tokens


class PositionalEncoding2D(nn.Module):
    """
    2D 正弦位置编码（用于图像特征）
    """
    
    def __init__(self, embed_dim=1024, temperature=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.temperature = temperature
    
    def forward(self, H, W):
        """
        Args:
            H, W: 特征图的高度和宽度
        
        Returns:
            pos_encoding: [H*W, embed_dim]
        """
        # 生成网格坐标
        y_embed = torch.arange(H, dtype=torch.float32).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, dtype=torch.float32).unsqueeze(0).repeat(H, 1)
        
        # 归一化到 [0, 1]
        y_embed = y_embed / H
        x_embed = x_embed / W
        
        # 生成频率
        dim_t = torch.arange(self.embed_dim // 2, dtype=torch.float32)
        dim_t = self.temperature ** (2 * dim_t / self.embed_dim)
        
        # Sin/Cos 编码
        pos_x = x_embed[: , : , None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        
        pos_x = torch.stack(
            [pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()], dim=3
        ).flatten(2)
        pos_y = torch. stack(
            [pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()], dim=3
        ).flatten(2)
        
        pos = torch.cat([pos_y, pos_x], dim=2).flatten(0, 1)  # [H*W, embed_dim]
        
        return pos



    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # --- 1. 基础维度与参数配置 ---
        self.joint_rep_type = cfg.MODEL.SMAL_HEAD.get('JOINT_REP', '6d')
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (cfg.SMAL.NUM_JOINTS + 1)
        self.npose = npose
        self.input_is_mean_shape = cfg.MODEL.SMAL_HEAD.get('TRANSFORMER_INPUT', 'zero') == 'mean_shape'
        
        # expecting all ready projected to the same dimension, e.g., 1024 for VGGT backbone
        # Get context dimension from IN_CHANNELS (e.g., 2048 for VGGT backbone)
        context_dim = cfg.MODEL.SMAL_HEAD.get('IN_CHANNELS', 1024)
        # self.bio_dim = cfg.MODEL.get('BIOCLIP_DIM', 512) 
        
        # --- 2. Coarse 阶段：生物先验模块 ---
        # 初始形状预测器 (从 BioCLIP 映射到 41 维 SMAL 形状)
        self.bio_to_betas_init = nn.Sequential(
            nn.Linear(context_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 41)
        )
        
        # 家族分类分支：用于校准 BioCLIP 空间
        # self.family_classifier = nn.Linear(context_dim, 5) 
        
        # 对比投影分支：将形状特征映射到 128 维空间进行 Contrastive Loss
        self.shape_projector = nn.Sequential(
            nn.Linear(41, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )
        
        # Bio-Context 投影：让 BioCLIP 信息作为 Token 进入 Transformer --no need anymore
        # self.bio_to_context = nn.Linear(self.bio_dim, context_dim)

        # --- 3. Fine 阶段：Transformer 迭代模块 ---
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 41 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
            # context_dim=context_dim,
        )
        transformer_args = {**transformer_args, **dict(cfg.MODEL.SMAL_HEAD.TRANSFORMER_DECODER)}
        
        # 假设已存在 TransformerDecoder 类
        self.transformer = TransformerDecoder(**transformer_args)
        
        dim = transformer_args['dim']
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 41)
        self.deccam = nn.Linear(dim, 3)

        # 初始化 Buffer
        self.register_buffer('init_pose', torch.zeros(1, npose))
        self.register_buffer('init_cam', torch.tensor([[0.9, 0, 0]]))
        self.register_buffer('init_betas_fallback', torch.zeros(1, 41))

    def forward(self, x, family_idx=None, **kwargs):
        
        bio_token = x[:,-1,:] # BioCLIP token 是最后一个 token , x already been augmented with bio token in the backbone, so we can directly take it out here.
        
        batch_size = x.shape[0]
        
        # --- 数据预处理 ---
        if len(x.shape) == 4:
            x = einops.rearrange(x, 'b c h w -> b (h w) c') 

        # --- [STEP 1: Coarse] 生物先验校准 ---
        # 1. 预测初始形状 (作为 IEF 的起点)
        init_betas = self.bio_to_betas_init(bio_token) # (B, 41)
        
        # 2. 准备监督信号：分类 Logits 和 对比特征
        # family_logits = self.family_classifier(bio_token) # (B, 5)
        shape_feat = F.normalize(self.shape_projector(init_betas), dim=1) # (B, 128)
        
        # 3. 准备增强的 Context (Image Tokens + Bio Token)
        # bio_token = self.bio_to_context(bio_embed).unsqueeze(1)
        # augmented_context = torch.cat([bio_token, x], dim=1)
        # --- [STEP 2: Fine] IEF 迭代修正 ---
        pred_pose = self.init_pose.expand(batch_size, -1)
        pred_cam = self.init_cam.expand(batch_size, -1)
        pred_betas = init_betas # 起点不再是 0，而是 Bio-Prior

        pred_pose_list, pred_betas_list, pred_cam_list = [], [], []

        for i in range(self.cfg.MODEL.SMAL_HEAD.get('IEF_ITERS', 3)):
            if self.input_is_mean_shape:
                token = torch.cat([pred_pose, pred_betas, pred_cam], dim=1)[:, None, :]
            else:
                token = torch.zeros(batch_size, 1, 1).to(x.device)

            token_out = self.transformer(token, context=x)
            token_out = token_out.squeeze(1)

            # 残差更新
            pred_pose = self.decpose(token_out) + pred_pose
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam

            pred_pose_list.append(pred_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        # --- 参数转换与封装 ---
        # (假设 joint_conversion_fn 已定义，将 6D 或 AA 转为 RotMat)
        # ... 后处理代码与原版相同 ...
        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        pred_smal_params_list = {}
        pred_smal_params_list['pose'] = torch.cat(
            [joint_conversion_fn(pbp).view(batch_size, -1, 3, 3)[:, 1:, :, :] for pbp in pred_pose_list], dim=0)
        pred_smal_params_list['betas'] = torch.cat(pred_betas_list, dim=0)
        pred_smal_params_list['cam'] = torch.cat(pred_cam_list, dim=0)
        pred_pose = joint_conversion_fn(pred_pose).view(batch_size, self.cfg.SMAL.NUM_JOINTS + 1, 3, 3)

        pred_smal_params = {'global_orient': pred_pose[:, [0]],
                            'pose': pred_pose[:, 1:],
                            'betas': pred_betas,
                            }
        extra_outputs = {
            # 'family_logits': family_logits,
            'shape_feat': shape_feat,
            'init_betas': init_betas
        }
        
        return pred_smal_params, pred_cam, extra_outputs