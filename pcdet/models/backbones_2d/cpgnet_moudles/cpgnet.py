import collections

import torch.nn as nn

from .. import cpgnet_moudles

def _list(x):
    if isinstance(x, collections.abc.Iterable):
        return list(x)
    return [x]

class Encoder(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.UNet = nn.ModuleList()
        self.UNet.append(
            getattr(cpgnet_moudles, model_cfg['NAME'])(**model_cfg['ARGS'])
        )

    def forward(self, x):
        encoding_features = {}
        for model in self.UNet:
            x, features = model(x)
            encoding_features.update(features)
        return x, encoding_features
    