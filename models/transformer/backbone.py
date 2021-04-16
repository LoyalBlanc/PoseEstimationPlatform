import math

import torch
import torch.nn.functional as f
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, output_dim, name='resnet18',
                 pretrained=True, return_inter_layers=True):
        super(BackboneBase, self).__init__()

        self.norm = nn.LayerNorm([3, 256, 256])
        backbone = getattr(torchvision.models, name)

        if return_inter_layers:
            return_layers = {'layer2': "1", 'layer3': "2", 'layer4': "3"}
            self.strides = [8, 16, 32]
            self.num_channels = [128, 256, 512] \
                if name in ('resnet18', 'resnet34') else [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [512] if name in ('resnet18', 'resnet34') else [2048]

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channel, output_dim, kernel_size=1),
                nn.GroupNorm(32, output_dim),
            ) for channel in self.num_channels
        ])

        self.additional_proj = nn.Sequential(
            nn.Conv2d(self.num_channels[-1], output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(32, output_dim),
        )
        self.body = IntermediateLayerGetter(
            backbone(pretrained=pretrained, norm_layer=FrozenBatchNorm2d),
            return_layers=return_layers)

    def forward(self, x):
        x = self.norm(x)
        out = []
        for index, (name, res) in enumerate(self.body(x).items()):
            out.append(self.input_proj[index](res))
        out.append(self.additional_proj(res))
        return out


class PositionEmbeddingSine(nn.Module):
    def __init__(self, input_dim, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = input_dim // 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask):
        assert mask is not None
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Backbone(nn.Module):
    def __init__(self, output_dim, name='resnet18', pretrained=True, scale=None):
        super(Backbone, self).__init__()
        self.backbone = BackboneBase(output_dim, name=name, pretrained=pretrained)
        self.pos_embed = PositionEmbeddingSine(output_dim, scale=scale)

    def forward(self, img, mask):
        src_list = self.backbone(img)
        mask_list = [f.interpolate(mask[None].float(), size=src.shape[-2:])[0].bool() for src in src_list]
        pos_list = [self.pos_embed(~mark) for mark in mask_list]
        return src_list, mask_list, pos_list


if __name__ == "__main__":
    seq_length = 7
    test_image = torch.randn(seq_length, 3, 256, 256)
    test_mask = torch.randn(seq_length, 256, 256)

    model = Backbone(128, name='resnet18', pretrained=False)

    s_list, m_list, p_list = model(test_image, test_mask)
    for s, m, p in zip(s_list, m_list, p_list):
        print(s.shape, m.shape, p.shape)
