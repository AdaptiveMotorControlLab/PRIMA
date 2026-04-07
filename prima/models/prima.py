import torch
import pickle
import pytorch_lightning as pl
from typing import Any, Dict
from yacs.config import CfgNode

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.utils import make_grid
from ..utils.geometry import perspective_projection, aa_to_rotmat
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_smal_head
from ..utils import MeshRenderer
from ..utils import renderer
from prima.models.smal_wrapper import SMAL
from .discriminator import Discriminator

from .bioclip_embedding import BioClipEmbedding
import sys
from transformers import AutoModel, AutoFeatureExtractor
import einops

import open_clip


from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss, ShapePriorLoss, PosePriorLoss, SupConLoss
log = get_pylogger(__name__)


class PRIMA(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup PRIMA model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg
        # Create backbone feature extractor

        if cfg.MODEL.BACKBONE.TYPE =='vith':
            self.backbone = create_backbone(cfg) # create vit backbone anyway, for inference, no config loading, just load ckpt weights 
        
            if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None): # pretrained exists and not none, then true
                
                log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
                state_dict = torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu', weights_only=True)['state_dict']
                state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
                
                missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
        
   
        
        elif cfg.MODEL.BACKBONE.TYPE =='dinov3':
            log.info(f'Loading DINOv3 backbone weights from torch hub') 
            # load  feature extractor

            self.backbone = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-lvd1689m")
            for param in self.backbone.parameters():
                param.requires_grad = True
            

            
        # freeze backbones (only dino or dino + aa)
        if cfg.MODEL.BACKBONE.get('FREEZE', False) and cfg.MODEL.BACKBONE.FREEZE in ['dinov2', 'dinov3']:
            log.info(f'Freezing only dino backbones parameters')
            for p in self.backbone.parameters(): # freeze dino backbone  
                p.requires_grad = False
        elif cfg.MODEL.BACKBONE.get('FREEZE', False) and cfg.MODEL.BACKBONE.TYPE == 'vith':
            log.info(f'Freezing first 2/3 blocks of vit backbone')
            # Freeze patch embedding
            if hasattr(self.backbone, 'patch_embed'):
                for p in self.backbone.patch_embed.parameters():
                    p.requires_grad = False
            
            # Freeze first 2/3 of transformer blocks
            if hasattr(self.backbone, 'blocks'):
                total_blocks = len(self.backbone.blocks)
                freeze_blocks = int(total_blocks * 2 / 3)
                log.info(f'Freezing {freeze_blocks} out of {total_blocks} blocks')
                for i in range(freeze_blocks):
                    for p in self.backbone.blocks[i].parameters():
                        p.requires_grad = False

        # Create SMAL head (predicts SMAL params + perspective camera)
        self.smal_head = build_smal_head(cfg)

        # Instantiate SMAL model
        smal_model_path = cfg.SMAL.MODEL_PATH
        with open(smal_model_path, 'rb') as f:
            smal_cfg = pickle.load(f, encoding="latin1")
        self.smal = SMAL(**smal_cfg)

        # create bioclip model for species classification token extraction 
        use_bioclip_embedding = cfg.MODEL.get('USE_BIOCLIP_EMBEDDING', False)
        if use_bioclip_embedding:
            bioclip_config = cfg.MODEL.get('BIOCLIP_EMBEDDING', {})
            embed_dim = bioclip_config.get('EMBED_DIM', 1280)
            self.bioclip_embedding = BioClipEmbedding(cfg, embed_dim=embed_dim)
            # Freeze BioClip model by default
            for param in self.bioclip_embedding.species_model.parameters():
                param.requires_grad = False
        else:
            self.bioclip_embedding = None
        
        # Create discriminator
        self.discriminator = Discriminator()
        

                

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        
        if self.cfg.LOSS_WEIGHTS.get('INTERMEDIATE_KP2D', 0) > 0:
            self.intermediate_kp2d_loss = Keypoint2DLoss(loss_type='l1')
        if self.cfg.LOSS_WEIGHTS.get('INTERMEDIATE_KP3D', 0) > 0:
            self.intermediate_kp3d_loss = Keypoint3DLoss(loss_type='l1')
        self.smal_parameter_loss = ParameterLoss()
        self.shape_prior_loss = ShapePriorLoss(path_prior=cfg.SMAL.SHAPE_PRIOR_PATH)
        self.pose_prior_loss = PosePriorLoss(path_prior=cfg.SMAL.POSE_PRIOR_PATH)
        self.supcon_loss = SupConLoss()


        self.register_buffer('initialized', torch.tensor(False))

        # init depth renderer for supervised training 
        # Setup renderer for visualization
        if init_renderer:
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.smal.faces.numpy())
        else:
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

    def get_parameters(self):
        all_params = list(self.smal_head.parameters())
        if self.cfg.MODEL.BACKBONE.TYPE in ['vith', 'dinov2', 'dinov3']:
            all_params += list(self.backbone.parameters())
        elif self.cfg.MODEL.BACKBONE.TYPE in ['concat','bb']:
            all_params += list(self.vit_backbone.parameters())
            all_params += list(self.dino_backbone.parameters())
            if hasattr(self, 'aggregator'):
                all_params += list(self.aggregator.parameters())
            if hasattr(self, 'fusion'):
                all_params += list(self.fusion.parameters())     


        if hasattr(self, 'keypoint_projection') and self.keypoint_projection is not None:
            all_params += list(self.keypoint_projection.parameters())
        if hasattr(self, 'bioclip_embedding') and self.bioclip_embedding is not None:
            # Only add projection parameters as the model itself is frozen
            all_params += list(self.bioclip_embedding.projection.parameters())
        return all_params

    def configure_optimizers(self):
        """
        Setup model and discriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        # Use separate learning rates only for vith backbone
        if self.cfg.MODEL.BACKBONE.TYPE == 'vith':
            # Separate backbone parameters and other parameters
            backbone_params = []
            other_params = []
            
            # Collect backbone parameters
            if hasattr(self, 'backbone'):
                backbone_params = list(filter(lambda p: p.requires_grad, self.backbone.parameters()))
            
            # Collect other parameters
            other_params += list(self.smal_head.parameters())


            if hasattr(self, 'keypoint_projection') and self.keypoint_projection is not None:
                other_params += list(self.keypoint_projection.parameters())
            if hasattr(self, 'bioclip_embedding') and self.bioclip_embedding is not None:
                other_params += list(self.bioclip_embedding.projection.parameters())

            
            # Filter only trainable parameters
            other_params = list(filter(lambda p: p.requires_grad, other_params))
            
            # Create parameter groups with different learning rates
            param_groups = [
                {'params': backbone_params, 'lr': self.cfg.TRAIN.LR / 10.0},  # Backbone: 1/10 lr
                {'params': other_params, 'lr': self.cfg.TRAIN.LR}  # Other modules: normal lr
            ]
            
            log.info(f'Using separate LR for vith backbone')
            log.info(f'Backbone parameters: {len(backbone_params)}, lr={self.cfg.TRAIN.LR / 10.0}')
            log.info(f'Other parameters: {len(other_params)}, lr={self.cfg.TRAIN.LR}')
        else:
            # Use same learning rate for all parameters
            all_params = list(filter(lambda p: p.requires_grad, self.get_parameters()))
            param_groups = [{'params': all_params, 'lr': self.cfg.TRAIN.LR}]
            log.info(f'Using same LR for all parameters: {len(all_params)}, lr={self.cfg.TRAIN.LR}')
        
        optimizer = torch.optim.AdamW(params=param_groups,
                                      weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        if self.cfg.LOSS_WEIGHTS.get("ADVERSARIAL", 0) > 0:
            optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                               lr=self.cfg.TRAIN.LR,
                                               weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        else:
            return optimizer,

        return optimizer, optimizer_disc

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        x = batch['img']  # [B, 3, H, W]
        batch_size = x.shape[0]

        # Compute conditioning features using the backbone
        if self.cfg.MODEL.BACKBONE.TYPE =='vith': # vit backbone return [1, 1280, 12, 16]
            conditioning_feats, cls = self.backbone(x[:, :, :, 32:-32])  #  reshape the input into [256, 192]
            # return shape shape [B, D, Hp, Wp], [B, D]
            if conditioning_feats.ndim == 4:
                # Flatten spatial dimensions into sequence dimension: [B, D, Hp, Wp] -> [B, Hp*Wp, D]
                B, D, Hp, Wp = conditioning_feats.shape
                conditioning_feats = conditioning_feats.permute(0, 2, 3, 1).reshape(B, Hp * Wp, D)  # [B, Hp*Wp, D]
            
            
        elif self.cfg.MODEL.BACKBONE.TYPE =='dinov2': # dino backbone, return [1, 329, 1024] 18*18+5=329.  kernel size dinov2=14
            bb_output = self.backbone(x[:, :, :, :]) # dino use square ? yes # what the output of dino
            conditioning_feats, cls = bb_output["x_norm_patchtokens"],bb_output['x_norm_clstoken']  #[1,324,1024], [1,1,1024]
            cls = cls.squeeze(1) # into [B,1024]
            
            
            
        elif self.cfg.MODEL.BACKBONE.TYPE == 'dinov3': # dino backbone, return [1, 329, 1024] 18*18+5=329.  kernel size dinov2=14; dinov3=16
            self.backbone.train(self.training)
            bb_output = self.backbone(x[:, :, :, :])
            conditioning_feats, cls = bb_output["last_hidden_state"][:,5:,:],bb_output['pooler_output']
            

        
  
        # add bioclip embedding if enabled
        if self.bioclip_embedding is not None:
            species_feature = self.bioclip_embedding(batch['img'])  # [B, embed_dim]

            # concatenate species feature to conditioning_feats along token dimension
            if len(conditioning_feats.shape) == 3:
                # Token-wise concatenation: add species_feature as a single token
                # (B, embed_dim) -> (B, 1, embed_dim)
                species_token = species_feature.unsqueeze(1)  # (B, 1, embed_dim)
                # Concatenate along token dimension: (B, num_tokens, C) + (B, 1, embed_dim) -> (B, num_tokens + 1, C or embed_dim)
                # Note: This requires C == embed_dim for consistent feature dimensions
                conditioning_feats = torch.cat([conditioning_feats, species_token], dim=1)  # (B, num_tokens + 1, C)
            else:
                # If conditioning_feats is 2D (B, C), concat directly along feature dimension
                conditioning_feats = torch.cat([conditioning_feats, species_feature], dim=-1)
        
        # Predict SMAL parameters and camera
        pred_smal_params, pred_cam, extra_outputs = self.smal_head(conditioning_feats)
        
        
        # Store useful regression outputs to the output dict
        output = {}
        
        if 'shape_feat' in extra_outputs:
            output['shape_feat'] = extra_outputs['shape_feat']
        
        if 'init_betas' in extra_outputs:
            output['init_betas'] = extra_outputs['init_betas'].reshape(batch_size, -1)
        

        output['pred_cam'] = pred_cam  # [B, 3]
        output['pred_smal_params'] = {k: v.clone() for k, v in pred_smal_params.items()}
        
    

        # Compute camera translation
        focal_length = batch['focal_length']
        
        pred_cam_t = torch.stack([
            pred_cam[:, 1],
            pred_cam[:, 2],
            2 * focal_length[:, 0] / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9)
        ], dim=-1)  # [B, 3]
        
        output['pred_cam_t'] = pred_cam_t  # [B, 3]
        output['focal_length'] = focal_length  # [B, 2]

        # Compute model vertices, joints and the projected joints
        pred_smal_params['global_orient'] = pred_smal_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_smal_params['pose'] = pred_smal_params['pose'].reshape(batch_size, -1, 3, 3)
        pred_smal_params['betas'] = pred_smal_params['betas'].reshape(batch_size, -1)
        smal_output = self.smal(**pred_smal_params, pose2rot=False)
        
        pred_keypoints_3d = smal_output.joints
        pred_vertices = smal_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        
        # project 3D keypoints to 2D
        pred_keypoints_2d = perspective_projection(
            pred_keypoints_3d,
            translation=pred_cam_t,
            focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE
        )  # [B, num_joints, 2]
        output['pred_keypoints_2d'] = pred_keypoints_2d
        
        # get intermediate keypoint predictions if available

        if 'keypoints_3d' in pred_smal_params and pred_smal_params['keypoints_3d'] is not None:
            inter_keypoints_3d = pred_smal_params['keypoints_3d']
            output['inter_keypoints_3d'] = inter_keypoints_3d.reshape(batch_size, -1, 3)
            # output['use_intermediate_kp3d_loss'] = True
        
        if 'keypoints_2d' in pred_smal_params and pred_smal_params['keypoints_2d'] is not None:
            inter_keypoints_2d = pred_smal_params['keypoints_2d']
            output['inter_keypoints_2d'] = inter_keypoints_2d.reshape(batch_size, -1, 2)
            # output['use_intermediate_kp2d_loss'] = True

        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """
        
        pred_smal_params = output['pred_smal_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']
        
        if 'inter_keypoints_2d' in output:
            inter_keypoints_2d = output['inter_keypoints_2d']
        if 'inter_keypoints_3d' in output:
            inter_keypoints_3d = output['inter_keypoints_3d']

        batch_size = pred_smal_params['pose'].shape[0]
        device = pred_smal_params['pose'].device
        dtype = pred_smal_params['pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_smal_params = batch['smal_params']
        gt_mask = batch['mask']
        has_smal_params = batch['has_smal_params']
        is_axis_angle = batch['smal_params_is_axis_angle']
        has_mask = batch['has_mask']
        
        # Compute 2D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        
        # Compute 3D keypoint loss
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)
        
        # Compute intermediate 2D keypoint loss if available
        loss_intermediate_kp2d = torch.tensor(0., device=device, dtype=dtype)
        if 'inter_keypoints_2d' in output:
            loss_intermediate_kp2d = self.intermediate_kp2d_loss(inter_keypoints_2d, gt_keypoints_2d)
            # loss_keypoints_2d = loss_keypoints_2d + loss_intermediate_kp2d

        # Compute intermediate 3D keypoint loss if available
        loss_intermediate_kp3d = torch.tensor(0., device=device, dtype=dtype)
        if 'inter_keypoints_3d' in output:
            loss_intermediate_kp3d = self.intermediate_kp3d_loss(inter_keypoints_3d, gt_keypoints_3d, pelvis_id=0)
            # loss_keypoints_3d = loss_keypoints_3d + loss_intermediate_kp3d
        
        # add intermediate keypoint losses if available

        # Compute loss on SMAL parameters
        loss_smal_params = {}
        for k, pred in pred_smal_params.items():
            # Skip keypoint predictions - they're handled separately
            if k in ['keypoints_2d', 'keypoints_3d']:
                continue
                
            gt = gt_smal_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_smal_params[k]
            
            # Only compute parameter loss if ANY sample has GT
            param_loss = self.smal_parameter_loss(pred.reshape(batch_size, -1),
                                                   gt.reshape(batch_size, -1),
                                                   has_gt)
            
            if k == "betas":
                # Only add shape prior loss if NOT all samples have GT (prior is regularization for samples without GT)
                # But the shape_prior_loss already handles this check internally
                loss_smal_params[k] = param_loss + self.shape_prior_loss(pred, batch["category"], has_gt)
                if 'init_betas' in output:
                    init_betas = output['init_betas']
                    loss_smal_params[k] = loss_smal_params[k] + self.shape_prior_loss(init_betas, batch["category"], has_gt) / 2.
                
            else:
                # Only add pose prior loss if NOT all samples have GT
                # The pose_prior_loss already handles this check internally
                loss_smal_params[k] = param_loss + \
                                      self.pose_prior_loss(torch.cat((pred_smal_params["global_orient"],
                                                                      pred_smal_params["pose"]),
                                                                      dim=1), has_gt) / 2.
        if 'shape_feat' in output:
            loss_supcon = self.supcon_loss(output['shape_feat'], labels=batch['category'])
        else: 
            loss_supcon = torch.tensor(0., device=device, dtype=dtype)
        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d + \
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d + \
               sum([loss_smal_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_smal_params]) + \
               self.cfg.LOSS_WEIGHTS['SUPCON'] * loss_supcon
        
        if 'inter_keypoints_2d' in output:
            loss = loss + self.cfg.LOSS_WEIGHTS.get('INTERMEDIATE_KP2D', 0) * loss_intermediate_kp2d
        if 'inter_keypoints_3d' in output:
            loss = loss + self.cfg.LOSS_WEIGHTS.get('INTERMEDIATE_KP3D', 0) * loss_intermediate_kp3d


        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach(),
                      loss_supcon=loss_supcon.detach(),
                      )

        for k, v in loss_smal_params.items():
            losses['loss_' + k] = v.detach()
            
        # attach intermediate keypoint losses if computed
        if 'inter_keypoints_2d' in output:
            losses['loss_inter_keypoints_2d'] = loss_intermediate_kp2d.detach()
        if 'inter_keypoints_3d' in output:
            losses['loss_inter_keypoints_3d'] = loss_intermediate_kp3d.detach()
            


        output['losses'] = losses

        return loss
    
    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            pose (torch.Tensor): Regressed pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = pose.shape[0]
        gt_pose = batch['pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_pose.view(-1, 3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat.detach(), gt_betas.detach())
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()    

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True,
                            write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        
        images = batch['img']
        gt_keypoints_2d = batch['keypoints_2d']
        batch_size = images.shape[0]
        
        # mul std then add mean
        images = (images) * (torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1, 3, 1, 1))
        images = (images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1, 3, 1, 1))

        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode + '/' + loss_name, val.detach().item(), step_count)
            # if train is False:
            #     for metric_name, val in output['metric'].items():
            #         summary_writer.add_scalar(mode + '/' + metric_name, val, step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               self.cfg.SMAL.get("FOCAL_LENGTH", 1000),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               pred_masks=output.get('pred_masks', None)[:num_images] if output.get('pred_masks', None) is not None else None,
                                                               gt_masks=output.get('gt_masks', None)[:num_images] if output.get('gt_masks', None) is not None else None,
                                                               )
        predictions = make_grid(predictions, nrow=5, padding=2)
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions

    def training_step(self, batch: Dict) -> Dict:
        """
        Run a full training step
        Args:
            batch (Dict): Dictionary containing {'img', 'mask', 'keypoints_2d', 'keypoints_3d', 'orig_keypoints_2d',
                                                'box_center', 'box_size', 'img_size', 'smal_params',
                                                'smal_params_is_axis_angle', '_trans', 'imgname', 'focal_length'}
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = batch['img']
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.get("ADVERSARIAL", 0) > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_smal_params = output['pred_smal_params']
        loss = self.compute_loss(batch, output, train=True)
        if self.cfg.LOSS_WEIGHTS.get("ADVERSARIAL", 0) > 0:
            disc_out = self.discriminator(pred_smal_params['pose'].reshape(batch_size, -1),
                                          pred_smal_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL,
                                                error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # For compatibility
        # if self.cfg.LOSS_WEIGHTS.ADVERSARIAL == 0:
        #     optimizer.param_groups[0]['capturable'] = True
        
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.get("ADVERSARIAL", 0) > 0:
            loss_disc = self.training_step_discriminator(batch['smal_params'],
                                                         pred_smal_params['pose'].reshape(batch_size, -1),
                                                         pred_smal_params['betas'].reshape(batch_size, -1),
                                                         optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        # Log training loss to the logger so checkpoint callback can monitor it.
        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=batch_size, sync_dist=True)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # The validation dataloader yields the inner batch dict directly (not wrapped as {'img': loader}).
        # Run forward, compute loss and log aggregated validation metrics so ModelCheckpoint can monitor them.
        output = self.forward_step(batch, train=False)
        # compute_loss will populate output['losses'] and return the scalar loss
        loss = self.compute_loss(batch, output, train=False)

        # Ensure losses dict is available
        losses = output.get('losses', {})

        # Log all validation losses to logger with on_epoch=True so checkpoint monitors epoch-level metric
        for loss_name, val in losses.items():
            # use prog_bar only for the main loss
            prog = True if loss_name == 'loss' else False
            # Log as 'val/<loss_name>' e.g. 'val/loss'
            self.log(f'val/{loss_name}', val, on_step=False, on_epoch=True, prog_bar=prog, logger=True,
                     sync_dist=True)

        # Periodically write images/other visuals to tensorboard
        # Log visualizations on the first batch of each validation epoch
        if batch_idx == 0:
            # Use global_step for step count when logging validation visuals
            self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output