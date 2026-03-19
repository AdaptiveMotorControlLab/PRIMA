from .vit import vith




def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE in ['vith','concat','aa']:   # vit bb will be used in these three cases - animal feature extractor 
        return vith(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
