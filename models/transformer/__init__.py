from .backbone import Backbone
from .transformer import DeformableTransformer

from torch import nn


class PoseFormer(nn.Module):
    def __init__(self):
        super(PoseFormer, self).__init__()
        self.backbone = Backbone(256, name='resnet18', pretrained=True)
        self.transformer = DeformableTransformer(d_model=256, nhead=8,
                                                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024,
                                                 dropout=0.1,
                                                 activation="relu", return_intermediate_dec=False,
                                                 num_feature_levels=4, dec_n_points=4, enc_n_points=4)
