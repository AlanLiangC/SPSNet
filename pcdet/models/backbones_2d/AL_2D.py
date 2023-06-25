# from pcdet.models.backbones_2d import convmlp
import torch
import torch.nn as nn
from functools import partial
import numpy as np
from .cpgnet_moudles import CBN2d
from .cpgnet_moudles.fcn import DualDownSamplingBlock
from .cpgnet_moudles.fcnv2 import AttentionFeaturePyramidFusionV2
from . import AL_2D

norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=bias)
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out = out + identity
        out = self.relu(out)

        return out

class BasicBlock_CP(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, dilation, padding, stride = 1) -> None:
        super().__init__()
        self.act = nn.ReLU()
        self.bn = norm_fn(out_channels)
        self.conv = nn.Conv2d(in_channels=input_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                dilation=dilation,
                                padding=padding,
                                stride=stride)
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))
        

class EncBlock(nn.Module):
    def __init__(self,input_channels, range_view = False) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = int(input_channels * 2)
        self.conv1 = BasicBlock_CP(input_channels=self.input_channels,
                                out_channels=self.input_channels,
                                kernel_size=(3, 3),
                                dilation=1,
                                padding=1,
                                stride=1)

        self.conv2 = BasicBlock_CP(input_channels=self.input_channels,
                                out_channels=self.input_channels,
                                kernel_size=(3, 3),
                                dilation=2,
                                padding=2,
                                stride=1)

        self.conv3 = BasicBlock_CP(input_channels=self.input_channels,
                                out_channels=self.input_channels,
                                kernel_size=(2,2),
                                dilation=2,
                                padding=1,
                                stride=1)

        self.conv4 = BasicBlock_CP(input_channels=self.input_channels*3,
                                out_channels=self.output_channels,
                                kernel_size=(1, 1),
                                dilation=1,
                                padding=0,
                                stride=1)

        self.conv5 = BasicBlock_CP(input_channels=self.input_channels,
                                out_channels=self.output_channels,
                                kernel_size=(1, 1),
                                dilation=1,
                                padding=0,
                                stride=1)
        if range_view:
            self.pool = nn.AvgPool2d(kernel_size=(1,2),stride=(1,2))
        else:
            self.pool = nn.AvgPool2d(kernel_size=(2,2),stride=2)

    def forward(self,x):
        input_data = x
        output_1 = self.conv1(input_data) # [4, 64, 512, 512]
        output_2 = self.conv2(output_1) # [4, 64, 512, 512]
        output_3 = self.conv3(output_2) # [4, 64, 512, 512]

        output_123 = torch.cat([output_1,output_2,output_3], dim = 1) # [4, 192, 512, 512]

        output_123_1 = self.conv4(output_123) # [4, 128, 512, 512]
        output_1_1 = self.conv5(x) # [4, 128, 512, 512]

        output = output_123_1 + output_1_1

        output = self.pool(output) # [4, 128, 256, 256]

        return output

class DecBlock(nn.Module):
    def __init__(self, input_channels, range_view = False) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = input_channels // 2

        if range_view:
            self.transconv = nn.Sequential(nn.ConvTranspose2d(in_channels = self.input_channels,
                                                out_channels=self.output_channels,
                                                kernel_size=(3,3),
                                                padding=1,
                                                stride=(1,2),
                                                output_padding=(0,1)),
                                            norm_fn(self.output_channels),
                                            nn.ReLU())
        else:
            self.transconv = nn.Sequential(nn.ConvTranspose2d(in_channels = self.input_channels,
                                                out_channels=self.output_channels,
                                                kernel_size=(3,3),
                                                padding=1,
                                                stride=2,
                                                output_padding=1),
                                            norm_fn(self.output_channels),
                                            nn.ReLU())

        self.conv1 = BasicBlock_CP(input_channels=self.output_channels,
                                out_channels=self.output_channels,
                                kernel_size=(3,3),
                                dilation=1,
                                padding=1)

        self.conv2 = BasicBlock_CP(input_channels=self.output_channels,
                                out_channels=self.output_channels,
                                kernel_size=(3,3),
                                dilation=2,
                                padding=2)

        self.conv3 = BasicBlock_CP(input_channels=self.output_channels,
                                out_channels=self.output_channels,
                                kernel_size=(2,2),
                                dilation=2,
                                padding=1)

        self.conv4 = BasicBlock_CP(input_channels=self.output_channels*3,
                                out_channels=self.output_channels,
                                kernel_size=(1, 1),
                                dilation=1,
                                padding=0,
                                stride=1)

        self.conv5 = BasicBlock_CP(input_channels=self.output_channels,
                                out_channels=self.output_channels,
                                kernel_size=(1, 1),
                                dilation=1,
                                padding=0,
                                stride=1)

    def forward(self, x):
        ouput_1 = self.transconv(x) # [4, 64, 511, 511]
        output_2 = self.conv1(ouput_1) # [4, 64, 511, 511]
        output_3 = self.conv2(output_2) # [4, 64, 511, 511]
        output_4 = self.conv3(output_3) # [4, 64, 511, 511]
        output_234 = torch.cat([output_2, output_3, output_4], dim = 1) # [4, 192, 511, 511]
        output_234 = self.conv4(output_234) # [4, 64, 511, 511]
        output_1_1 = self.conv5(ouput_1) # [4, 64, 511, 511]
        output = output_234 + output_1_1

        return output
        

class CP_Unet(nn.Module): # layers_num = 4 in our project
    def __init__(self, input_channels, layers_num, output_channels, range_view = False):
        super().__init__()
        self.layers = [int(input_channels * 2**i) for i in range(layers_num)]

        self.pre_conv = BasicBlock(inplanes=input_channels, planes=input_channels)
        self.out_conv = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.encode_blocks = nn.ModuleList()
        self.decode_blocks = nn.ModuleList()
        self.basic_blocks = nn.ModuleList()

        for i in range(len(self.layers) - 1):
            self.encode_blocks.append(EncBlock(input_channels=self.layers[i], range_view = range_view))
            self.decode_blocks.append(DecBlock(input_channels=self.layers[0-1-i], range_view = range_view))
            self.basic_blocks.append(BasicBlock(inplanes=self.layers[-1-i], planes=self.layers[-2-i]))
    
    def forward(self,x):
        e0 = self.pre_conv(x) # [4, 16, 512, 512]
        
        e1 = self.encode_blocks[0](e0) # [4, 32, 256, 256]
        e2 = self.encode_blocks[1](e1) # [4, 64, 128, 128]
        e3 = self.encode_blocks[2](e2) # [4, 128, 64, 64]

        d0 = self.decode_blocks[0](e3) # [4, 64, 128, 128]
        d0 = torch.cat([e2, d0], dim = 1) # [4, 128, 128, 128]
        d0 = self.basic_blocks[0](d0) # [4, 64, 128, 128]

        d1 = self.decode_blocks[1](d0) # [4, 32, 256, 256]
        d1 = torch.cat([e1, d1], dim = 1) # [4, 64, 256, 256]
        d1 = self.basic_blocks[1](d1) # [4, 32, 256, 256]

        d2 = self.decode_blocks[2](d1) # [4, 16, 512, 512]
        d2 = torch.cat([e0, d2], dim = 1) # [4, 32, 512, 512]
        d2 = self.basic_blocks[2](d2) # [4, 16, 512, 512]

        out = self.out_conv(d2)

        out_dict = {}
        out_dict.update({
            'e1': e1,
            'e2': e2,
            'e3': e3,
            'd0': d0
        })

        return out, out_dict


class DetEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
        assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.UPSAMPLE_STRIDES), 'must have upsample process'

        layer_nums = self.model_cfg.LAYER_NUMS
        layer_strides = self.model_cfg.LAYER_STRIDES
        num_filters = self.model_cfg.NUM_FILTERS
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES

        num_levels = len(layer_nums) # normally is 2
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()

        for idx in range(num_levels):
            # downsample
            cur_c_out = num_filters[idx]
            cur_c_in = c_in_list[idx]
            cur_stride = layer_strides[idx]

            cur_layers = [
                nn.Conv2d(cur_c_in, cur_c_out, 3,
                    stride=cur_stride, padding=1, bias=False),
                norm_fn(cur_c_out),
                nn.ReLU()
            ]
            for _ in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(cur_c_out, cur_c_out, 3,
                              padding=1, bias=False),
                    norm_fn(cur_c_out),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

            # upsample
            cur_up_stride = upsample_strides[idx]
            cur_c_up_out= num_upsample_filters[idx]
            if cur_up_stride > 1:
                self.deblocks.append(nn.Sequential(
                    nn.ConvTranspose2d(cur_c_out, cur_c_up_out, 
                        cur_up_stride,stride=cur_up_stride,bias=False),
                    norm_fn(cur_c_up_out),
                    nn.ReLU(),
                ))
            else:
                cur_up_stride = np.round(1 / cur_up_stride).astype(np.int)
                self.deblocks.append(nn.Sequential(
                    nn.Conv2d(cur_c_out, cur_c_up_out, cur_up_stride,
                        stride=cur_up_stride, bias=False),
                    norm_fn(cur_c_up_out),
                    nn.ReLU(),
                ))
        c_in = sum(num_upsample_filters)
        self.num_bev_features = c_in

    def forward(self, x, **kwags):
        ups = []
        encoding_features = []
        encoding_features.append(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            encoding_features.append(x)
            ups.append(self.deblocks[i](x))
        x = torch.cat(ups, dim=1)
    
        return x, encoding_features



class DetDecoder(nn.Module):

    def __init__(self, 
                 in_channels=64, 
                 encoder_channels=[32, 64, 128, 128], 
                 stride=[2, 2, 2, 2],
                 decoder_channels=[96, 64, 64, 64]):
        super(DetDecoder, self).__init__()
        assert len(encoder_channels) == len(stride)
        assert len(decoder_channels) <= len(encoder_channels)


        self.decoder = nn.ModuleList()
        for lower_c, higher_c, out_c in zip(([in_channels]+encoder_channels)[:len(decoder_channels)][::-1],
                                             encoder_channels[-1:] + decoder_channels[:-1],
                                             decoder_channels):
            self.decoder.append(AttentionFeaturePyramidFusionV2(lower_c, higher_c, out_c))

    def forward(self, encoder_outputs):
        """
        Param:
            inputs: with shape of :math:`(N,C,H,W)`, where N is batch size
        """
        outputs = encoder_outputs[-1]
        for layer, inputs_lower in zip(self.decoder, encoder_outputs[: len(self.decoder)][::-1]):
            outputs = layer(inputs_lower, outputs)
            # viz.image(outputs[0,0,...].clamp(0,1), opts={"title": f'd{layer_idx}'})

        return outputs



class AL_Unet(nn.Module):
    def __init__(self,model_config):
        super().__init__()
        self.model_config = model_config
        self.de_config = model_config.DETENCODER
        self.dd_config = model_config.DETDECODER

        self.det_encoder = DetEncoder(self.de_config, self.de_config.INPUT_CHANNELS)
        self.det_decoder = getattr(AL_2D, self.dd_config['NAME'])(**self.dd_config['ARGS'])
        # self.det_decoder = DetDecoder(self.dd_config)
        
    # backbone_3d.bev_2D.det_decoder.decoder.0.cbn_higher.conv.weight
    def forward(self, x):
        det_feature_map, encoding_features = self.det_encoder(x)
        bev_seg_feature_map =self.det_decoder(encoding_features)

        return det_feature_map, bev_seg_feature_map






  

        

        





if __name__ == "__main__":
    data = torch.randn(4,64,512,512)
    # model = convmlp.convmlp_s()
    # model = EncBlock(input_channels=64)
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'Total number of params: {n_parameters}')
    # output = model(data)
    # print(output.shape)

    # model = DecBlock(input_channels = output.shape[1])
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'Total number of params: {n_parameters}')
    # output = model(output)
    # print(output.shape)

    model = CP_Unet(input_channels=64,layers_num=4,output_channels=64)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of params: {n_parameters}')
    output = model(data)
    print(output.shape)
