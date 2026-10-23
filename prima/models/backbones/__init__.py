"""
PRIMA: Boosting Animal Mesh Recovery with Taxonomic Priors and Adaptation-Driven Data Bootstrapping

Official implementation of the paper:
"PRIMA: Boosting Animal Mesh Recovery with Taxonomic Priors and Adaptation-Driven Data Bootstrapping"
by Xiaohang Yu, Ti Wang, and Mackenzie Weygandt Mathis
Licensed under a modified MIT license
"""

from .vit import vith




def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE in ['vith','concat','aa']:   # vit bb will be used in these three cases - animal feature extractor 
        return vith(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
