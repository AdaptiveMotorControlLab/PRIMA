import torch
import torch.nn as nn
import numpy as np
import pickle
from pytorch3d.transforms import matrix_to_axis_angle
import torch.nn.functional as F
from ..utils.geometry import aa_to_rotmat
from typing import Dict



class DepthLoss(nn.Module):
    """
    Depth loss between predicted SMAL vertices and GT SMAL vertices.
    Compares vertex Z (camera space) after applying camera translation.
    Only computes loss for samples that have valid GT SMAL parameters.
    """
    def __init__(self, loss_type: str = 'l1'):
        super().__init__()
        self.loss_type = loss_type
        self.l1 = nn.L1Loss(reduction='none')  # Changed to 'none' for per-sample masking
        self.l2 = nn.MSELoss(reduction='none')  # Changed to 'none' for per-sample masking

    def forward(self,
                pred_vertices: torch.Tensor,      # (B, V, 3)
                pred_cam_t: torch.Tensor,         # (B, 3)
                gt_smal_params: Dict[str, torch.Tensor],
                smal_model,                       # SMAL instance callable
                is_axis_angle: Dict[str, torch.Tensor],
                gt_cam_t: torch.Tensor = None,    # (B, 3) or None -> fallback to pred_cam_t
                has_smal_params: Dict[str, torch.Tensor] = None  # Added masking support
                ) -> torch.Tensor:
        batch_size = pred_vertices.shape[0]
        device = pred_vertices.device
        
        # Determine which samples have valid GT SMAL params
        # A sample is valid only if it has GT for pose, betas, and global_orient
        if has_smal_params is not None:
            valid_mask = (has_smal_params['pose'] * 
                         has_smal_params['betas'] * 
                         has_smal_params['global_orient']).bool()
            
            # If no samples have valid GT, return zero loss
            if valid_mask.sum() == 0:
                return torch.tensor(0., device=device, dtype=pred_vertices.dtype)
        else:
            # If not provided, assume all samples are valid
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # prepare GT params for SMAL
        gt_params_for_smal = {}
        for k in ['global_orient', 'pose', 'betas']:
            val = gt_smal_params[k].to(device=device)
            if k == 'betas':
                gt_params_for_smal[k] = val.view(batch_size, -1)
            else:
                gt_val = val.view(batch_size, -1)
                if is_axis_angle[k].all():
                    gt_val = aa_to_rotmat(gt_val.reshape(-1, 3)).view(batch_size, -1, 3, 3)
                else:
                    gt_val = gt_val.view(batch_size, -1, 3, 3)
                gt_params_for_smal[k] = gt_val

        # generate GT vertices (no grad)
        with torch.no_grad():
            gt_out = smal_model(**gt_params_for_smal, pose2rot=False)
            gt_vertices = gt_out.vertices.view(batch_size, -1, 3)

        if gt_cam_t is None:
            gt_cam_t = pred_cam_t

        # depth = z in camera coordinates
        pred_depth = (pred_vertices + pred_cam_t.unsqueeze(1))[..., 2]  # (B, V)
        gt_depth = (gt_vertices + gt_cam_t.unsqueeze(1))[..., 2]        # (B, V)

        # Compute loss per sample
        if self.loss_type == 'l1':
            loss_per_sample = self.l1(pred_depth, gt_depth).mean(dim=1)  # (B,)
        else:
            loss_per_sample = self.l2(pred_depth, gt_depth).mean(dim=1)  # (B,)
        
        # Apply mask: only compute loss for samples with valid GT
        masked_loss = loss_per_sample * valid_mask.float()
        
        # Return mean over valid samples
        num_valid = valid_mask.sum().float()
        if num_valid > 0:
            return masked_loss.sum() / num_valid
        else:
            return torch.tensor(0., device=device, dtype=pred_vertices.dtype)


class MaskLoss(nn.Module):
    """
    Mask loss between rendered predicted mesh mask and rendered GT mesh mask.
    This loss relies on a MeshRenderer-like object that provides `render_mask(vertices, camera_translation, focal_length)`
    returning a single-channel numpy mask (H, W) with values 0/1.
    """
    def __init__(self, mesh_renderer=None):
        super().__init__()
        self.mesh_renderer = mesh_renderer
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self,
                pred_vertices: torch.Tensor,      # (B, V, 3)
                pred_cam_t: torch.Tensor,         # (B, 3)
                gt_smal_params: Dict[str, torch.Tensor],
                smal_model,                       # SMAL instance callable
                is_axis_angle: Dict[str, torch.Tensor],
                gt_cam_t: torch.Tensor = None,    # optional (B,3)
                focal_length: float = 1000.0
                ) -> torch.Tensor:
        batch_size = pred_vertices.shape[0]
        device = pred_vertices.device

        # if no renderer available, return zero loss
        if self.mesh_renderer is None:
            return torch.tensor(0., device=device, dtype=pred_vertices.dtype)

        # prepare GT params for SMAL
        gt_params_for_smal = {}
        for k in ['global_orient', 'pose', 'betas']:
            val = gt_smal_params[k].to(device=device)
            if k == 'betas':
                gt_params_for_smal[k] = val.view(batch_size, -1)
            else:
                gt_val = val.view(batch_size, -1)
                if is_axis_angle[k].all():
                    gt_val = aa_to_rotmat(gt_val.reshape(-1, 3)).view(batch_size, -1, 3, 3)
                else:
                    gt_val = gt_val.view(batch_size, -1, 3, 3)
                gt_params_for_smal[k] = gt_val

        # generate GT vertices (no grad)
        with torch.no_grad():
            gt_out = smal_model(**gt_params_for_smal, pose2rot=False)
            gt_vertices = gt_out.vertices

        if gt_cam_t is None:
            gt_cam_t = pred_cam_t

        # convert to numpy for renderer
        pred_vertices_np = pred_vertices.detach().cpu().numpy()
        gt_vertices_np = gt_vertices.detach().cpu().numpy()
        cam_np = pred_cam_t.detach().cpu().numpy() if pred_cam_t is not None else np.zeros((batch_size, 3), dtype=np.float32)

        per_item_losses = []
        for i in range(batch_size):
            try:
                pred_mask = self.mesh_renderer.render_mask(pred_vertices_np[i], cam_np[i], focal_length)
                gt_mask_r = self.mesh_renderer.render_mask(gt_vertices_np[i], cam_np[i], focal_length)
                pm = torch.from_numpy(pred_mask).to(device=device, dtype=pred_vertices.dtype)
                gm = torch.from_numpy(gt_mask_r).to(device=device, dtype=pred_vertices.dtype)
                per_item_losses.append(self.l1(pm, gm))
            except Exception:
                # ignore render failure for this sample
                continue

        if len(per_item_losses) == 0:
            return torch.tensor(0., device=device, dtype=pred_vertices.dtype)

        return torch.stack(per_item_losses).mean()

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1, 2))
        return loss.sum()


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 0):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1, 2))
        return loss.sum()


class ParameterLoss(nn.Module):

    def __init__(self):
        """
        SMAL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor):
        """
        Compute SMAL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth MANO parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        mask = torch.ones_like(pred_param, device=pred_param.device, dtype=pred_param.dtype)
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims - 1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = (has_param * self.loss_fn(pred_param*mask, gt_param*mask))
        return loss_param.sum()


class PosePriorLoss(nn.Module):
    def __init__(self, path_prior):
        super(PosePriorLoss, self).__init__()
        with open(path_prior, "rb") as f:
            data_prior = pickle.load(f, encoding="latin1")

        self.register_buffer("mean_pose", torch.from_numpy(data_prior["mean_pose"]).float())
        self.register_buffer("precs", torch.from_numpy(np.array(data_prior["pic"])).float())

        use_index = np.ones(105, dtype=bool)
        use_index[:3] = False  # global rotation set False
        self.register_buffer("use_index", torch.from_numpy(use_index).float())

    def forward(self, x, has_gt):
        """
        Args:
            x: (batch_size, 35, 3, 3)
            has_gt: has pose?
        Returns:
            pose prior loss
        """
        if has_gt.sum() == len(has_gt):
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        has_gt = has_gt.type(torch.bool)
        x = x[~has_gt]
        x = matrix_to_axis_angle(x.reshape(-1, 3, 3))
        delta = x.reshape(-1, 35*3) - self.mean_pose
        loss = torch.tensordot(delta, self.precs, dims=([1], [0])) * self.use_index
        return (loss ** 2).mean()


class ShapePriorLoss(nn.Module):
    def __init__(self, path_prior):
        super(ShapePriorLoss, self).__init__()
        with open(path_prior, "rb") as f:
            data_prior = pickle.load(f, encoding="latin1")

        model_covs = np.array(data_prior["cluster_cov"])  # shape: (5, 41, 41)
        inverse_covs = np.stack(
            [np.linalg.inv(model_cov + 1e-5 * np.eye(model_cov.shape[0])) for model_cov in model_covs],
            axis=0)
        prec = np.stack([np.linalg.cholesky(inverse_cov) for inverse_cov in inverse_covs], axis=0)

        self.register_buffer("betas_prec", torch.FloatTensor(prec))
        self.register_buffer("mean_betas", torch.FloatTensor(data_prior["cluster_means"]))

    def forward(self, x, category, has_gt):
        """
        Args:
            x: predicted betas (batch_size, 41)
            category: animal category (batch_size,)
            has_gt: has shape?
        Returns:
            shape prior loss
        """
        if has_gt.sum() == len(has_gt):
            return torch.tensor(0.0, dtype=x.dtype, device=x.device)
        has_gt = has_gt.type(torch.bool)
        x, category = x[~has_gt], category[~has_gt]
        delta = (x - self.mean_betas[category.long()])  # [batch_size, 41]
        loss = []
        for x0, c0 in zip(delta, category):
            loss.append(torch.tensordot(x0, self.betas_prec[c0], dims=([0], [0])))
        loss = torch.stack(loss, dim=0)
        return (loss ** 2).mean()



class PrototypeSupConLoss(nn.Module):
    def __init__(self, prototypes_init, feat_dim=128, temperature=0.1):
        """
        prototypes_init: 预计算好的 (5, 512) BioCLIP 中心
        feat_dim: 你投影后的 shape_feat 维度 (128)
        """
        super().__init__()
        self.temperature = temperature
        
        # 这里的 prototypes 应该映射到你投影后的空间
        # 建议在训练开始时，先将 BioCLIP 中心通过一次你的 Projector 得到初始 Prototype
        self.register_buffer("prototypes", torch.randn(5, feat_dim))
        
    def forward(self, features, labels):
        """
        features: (B, 128) - 已经 L2 归一化后的共享特征或 shape_feat
        labels: (B,) - 5 类家族索引
        """
        # 1. 确保特征是归一化的
        features = F.normalize(features, p=2, dim=1)
        # 2. 确保中心也是归一化的
        prototypes = F.normalize(self.prototypes, p=2, dim=1)
        
        # 3. 计算样本与 5 个中心的相似度 (B, 5)
        # 这本质上是将中心作为“权重”进行分类，但具备对比学习的温度控制
        logits = torch.matmul(features, prototypes.T) / self.temperature
        
        # 4. 计算 Cross Entropy
        # 它的物理意义是：拉近样本与所属家族中心，推开与其他 4 个家族中心
        loss = F.cross_entropy(logits, labels)
        
        return loss

    @torch.no_grad()
    def update_prototypes(self, features, labels, momentum=0.999):
        """
        可选：在训练过程中动量更新中心，使其缓慢适应 3D 任务
        """
        for i in range(5):
            mask = (labels == i)
            if mask.any():
                new_mean = features[mask].mean(dim=0)
                self.prototypes[i] = momentum * self.prototypes[i] + (1 - momentum) * new_mean


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        features = torch.stack((features, features), dim=1)
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

# 在你的 losses.py 或训练脚本中

class InterLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.use_intermediate_supervision = cfg.LOSS. get('USE_INTERMEDIATE_SUPERVISION', True)
        self.intermediate_weight = cfg. LOSS.get('INTERMEDIATE_WEIGHT', 0.5)
        
        # 2D keypoint loss
        self. keypoint_2d_loss = nn.MSELoss(reduction='none')
        
        # 3D keypoint loss
        self.keypoint_3d_loss = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, gt_data):
        """
        Args:
            predictions: 模型输出 (pred_smal_params, pred_cam, pred_smal_params_list)
            gt_data: dict 包含 GT 数据
                - 'keypoints_2d': [B, N, 3] (x, y, visibility)
                - 'keypoints_3d':  [B, N, 3] (x, y, z) 或 [B, N, 4] (x, y, z, confidence)
        """
        pred_smal_params, pred_cam, pred_smal_params_list = predictions
        
        losses = {}
        total_loss = 0.0
        
        # ========== 最终预测的监督 ==========
        # (这里需要通过 SMAL 模型前向得到关键点)
        # 假设你已经有了最终的 keypoints_2d 和 keypoints_3d
        
        # ========== 中间预测的监督 ==========
        if self.use_intermediate_supervision and pred_smal_params_list is not None:
            
            # 2D keypoints 监督
            if 'keypoints_2d' in pred_smal_params_list and pred_smal_params_list['keypoints_2d'] is not None:
                pred_kps_2d_all = pred_smal_params_list['keypoints_2d']
                # [B*num_iters, N, 2]
                
                gt_kps_2d = gt_data['keypoints_2d'][: , :, :2]  # [B, N, 2]
                gt_vis_2d = gt_data['keypoints_2d'][:, :, 2]   # [B, N]
                
                # 复制 GT 到每次迭代
                num_iters = pred_kps_2d_all.shape[0] // gt_kps_2d.shape[0]
                gt_kps_2d_repeated = gt_kps_2d.repeat(num_iters, 1, 1)  # [B*num_iters, N, 2]
                gt_vis_2d_repeated = gt_vis_2d.repeat(num_iters, 1)     # [B*num_iters, N]
                
                # 计算 loss (只对可见的关键点)
                loss_2d = self.keypoint_2d_loss(pred_kps_2d_all, gt_kps_2d_repeated)
                loss_2d = loss_2d. mean(dim=-1)  # [B*num_iters, N]
                loss_2d = (loss_2d * gt_vis_2d_repeated).sum() / (gt_vis_2d_repeated. sum() + 1e-6)
                
                losses['intermediate_keypoints_2d'] = loss_2d * self.intermediate_weight
                total_loss += losses['intermediate_keypoints_2d']
            
            # 3D keypoints 监督
            if 'keypoints_3d' in pred_smal_params_list and pred_smal_params_list['keypoints_3d'] is not None:
                pred_kps_3d_all = pred_smal_params_list['keypoints_3d']
                # [B*num_iters, N, 3]
                
                gt_kps_3d = gt_data['keypoints_3d'][: , :, :3]  # [B, N, 3]
                if gt_data['keypoints_3d'].shape[-1] == 4:
                    gt_conf_3d = gt_data['keypoints_3d'][:, :, 3]  # [B, N]
                else:
                    gt_conf_3d = torch.ones_like(gt_kps_3d[:, :, 0])  # 全部有效
                
                # 复制 GT
                num_iters = pred_kps_3d_all. shape[0] // gt_kps_3d.shape[0]
                gt_kps_3d_repeated = gt_kps_3d.repeat(num_iters, 1, 1)
                gt_conf_3d_repeated = gt_conf_3d.repeat(num_iters, 1)
                
                # 计算 loss
                loss_3d = self.keypoint_3d_loss(pred_kps_3d_all, gt_kps_3d_repeated)
                loss_3d = loss_3d.mean(dim=-1)  # [B*num_iters, N]
                loss_3d = (loss_3d * gt_conf_3d_repeated).sum() / (gt_conf_3d_repeated.sum() + 1e-6)
                
                losses['intermediate_keypoints_3d'] = loss_3d * self.intermediate_weight
                total_loss += losses['intermediate_keypoints_3d']
        
        # ...  其他 losses (pose, shape, 等) ...
        
        losses['total'] = total_loss
        return losses