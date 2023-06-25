import numpy as np
import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

class RB_Fusion(nn.Module):
    expansion = 1

    def __init__(self, model_cfg, input_channels):
        super(RB_Fusion, self).__init__()

        self.model_cfg = model_cfg
        bev_feature_dim = model_cfg.BEV_DIM
        range_feature_dim = model_cfg.RANGE_DIM

        self.channel_avg_func = nn.AdaptiveAvgPool2d((1,1))
        self.channel_max_func = nn.AdaptiveMaxPool2d((1,1))
        self.channel_ln = nn.Sequential(nn.Linear(in_features=(bev_feature_dim + range_feature_dim) * 2, 
                                                out_features=bev_feature_dim,
                                                bias=False),
                                        nn.ReLU(),
                                        nn.Dropout(0.2),
                                        nn.Linear(in_features=bev_feature_dim,
                                                out_features=bev_feature_dim + range_feature_dim))
                                                
        self.space_ln = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3,3), stride=1, padding=1))
        

        self.act = nn.Sigmoid()

        self.num_bev_features = bev_feature_dim + range_feature_dim
        self.bev_feature_dim = bev_feature_dim
        self.range_feature_dim = range_feature_dim

    def forward(self, batch_dict):
        x = batch_dict['spatial_features']
        
        bev_feature = x[:,:self.bev_feature_dim,...]
        range_feature = x[:,self.bev_feature_dim:,...]
        
        # bev
        bev_channel_avg = self.channel_avg_func(bev_feature).squeeze()
        bev_channel_max = self.channel_max_func(bev_feature).squeeze()

        bev_space_avg = torch.mean(bev_feature, dim = 1).unsqueeze(dim = 1)
        bev_space_max = torch.max(bev_feature, dim = 1)[0].unsqueeze(dim = 1)

        # range
        range_channel_avg = self.channel_avg_func(range_feature).squeeze()
        range_channel_max = self.channel_max_func(range_feature).squeeze()

        range_space_avg = torch.mean(range_feature, dim = 1).unsqueeze(dim = 1)
        range_space_max = torch.max(range_feature, dim = 1)[0].unsqueeze(dim = 1)

        # attention map
        channel_wise = torch.cat([bev_channel_avg, range_channel_avg, bev_channel_max, range_channel_max], dim = -1)
        space_wise = torch.cat([bev_space_avg, range_space_avg, bev_space_max, range_space_max], dim = 1)

        channel_wise = self.act(self.channel_ln(channel_wise)).unsqueeze(dim = -1).unsqueeze(dim = -1)
        space_wise = self.act(self.space_ln(space_wise))
        # attention_map = channel_wise * space_wise
        # attention_map = self.act(attention_map)

        # out = attention_map*x + x
        out = channel_wise * x
        out = space_wise * out

        out = out + x
        
        batch_dict['spatial_features_2d'] = out

        return batch_dict