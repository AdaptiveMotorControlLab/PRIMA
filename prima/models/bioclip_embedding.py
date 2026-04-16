"""
PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Biological Priors and Test-Time Adaptation"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""

"""
bioclip Embedding Module
Converts image batch to embeddings that can be concatenated with image features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BioClipEmbedding(nn.Module):
    """
    Embeds images into a feature space using BioClip model that can be combined with image features.
    
    Args:
        embed_dim: Output embedding dimension, should match the dimension of image features for concatenation
    """
    
    def __init__(self, cfg, embed_dim: int = 1024):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        import open_clip
        
        if cfg.MODEL.BIOCLIP_EMBEDDING.TYPE == 'bioclip2':
            print("[BioClipEmbedding] Using BioClip2 model from Hugging Face Hub")
            self.species_model, _,_ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        else:   
            self.species_model, _,_ = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        # tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')


        self.species_model.eval()
        
        # Get the output dimension from the model
        bioclip_output_dim = self.species_model.visual.output_dim
        
        # Project to target dimension
        self.projection = nn.Sequential(
            nn.Linear(bioclip_output_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (B, C, H, W) representing a batch of images
        Returns:
            Tensor of shape (B, embed_dim) representing the embedded features
        """
        # BioClip expects 224x224 input, resize if needed
        if images.shape[-2:] != (224, 224):
            images_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            images_resized = images
            
        with torch.no_grad():
            image_features = self.species_model.encode_image(images_resized)
        
        projected_features = self.projection(image_features)
        
        return projected_features