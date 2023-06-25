import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CBN2d
from .fcn import DualDownSamplingBlock
# from visdom import Visdom
# viz = Visdom(server='http://127.0.0.1', port=8097)

class AttentionFeaturePyramidFusionV2(nn.Module):

    def __init__(self, lower_channels, higher_channels, out_channels):
        super(AttentionFeaturePyramidFusionV2, self).__init__()

        self.cbn_higher = CBN2d(higher_channels, out_channels)
        self.cbn_lower = CBN2d(lower_channels, out_channels)
        self.attn = nn.Conv2d(2*out_channels, 2, 3, 1, 1)

    def forward(self, inputs_lower, inputs_higher):

        inputs_higher = F.interpolate(inputs_higher, inputs_lower.shape[-2:], mode='bilinear')

        outputs_higher = self.cbn_higher(inputs_higher)
        outputs_lower = self.cbn_lower(inputs_lower)

        attn_weight = torch.softmax(self.attn(torch.cat([outputs_higher, outputs_lower], dim=1)), dim=1)
        outputs = outputs_higher * attn_weight[:, 0:1, :, :] + outputs_lower * attn_weight[:, 1:, :, :]
        
        return outputs


class CPGFCNV2(nn.Module):

    def __init__(self, 
                 in_channels=64, 
                 encoder_channels=[32, 64, 128, 128], 
                 stride=[2, 2, 2, 2],
                 decoder_channels=[96, 64, 64, 64]):
        super(CPGFCNV2, self).__init__()
        assert len(encoder_channels) == len(stride)
        assert len(decoder_channels) <= len(encoder_channels)

        self.encoder = nn.ModuleList()
        for in_c, out_c, s in zip(([in_channels] + encoder_channels)[:-1], encoder_channels, stride):
            self.encoder.append(DualDownSamplingBlock(in_c, out_c, s))

        self.decoder = nn.ModuleList()
        for lower_c, higher_c, out_c in zip(([in_channels]+encoder_channels)[:len(decoder_channels)][::-1],
                                             encoder_channels[-1:] + decoder_channels[:-1],
                                             decoder_channels):
            self.decoder.append(AttentionFeaturePyramidFusionV2(lower_c, higher_c, out_c))

    def forward(self, inputs):
        """
        Param:
            inputs: with shape of :math:`(N,C,H,W)`, where N is batch size
        """
        layer_idx = 0
        encoding_features = {}
        encoder_outputs = [inputs]
        # viz.image(inputs[0,0,...].clamp(0,1), opts={"title": 'input'})
        for layer in self.encoder:
            layer_outputs = layer(encoder_outputs[-1])
            encoder_outputs.append(layer_outputs)
            encoding_features.update({
                f'e{layer_idx}': layer_outputs
            })
            # viz.image(layer_outputs[0,0,...].clamp(0,1), opts={"title": f'e{layer_idx}'})
            layer_idx += 1

        outputs = encoder_outputs[-1]
        layer_idx = 0
        for layer, inputs_lower in zip(self.decoder, encoder_outputs[: len(self.decoder)][::-1]):
            outputs = layer(inputs_lower, outputs)
            encoding_features.update({
                f'd{layer_idx}': outputs
            })
            # viz.image(outputs[0,0,...].clamp(0,1), opts={"title": f'd{layer_idx}'})
            layer_idx += 1

        return outputs, encoding_features
    