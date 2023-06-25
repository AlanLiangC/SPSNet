import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

from .common import CBN2d


class DualDownSamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(DualDownSamplingBlock, self).__init__()

        self.cbn1 = CBN2d(in_channels, out_channels, stride=stride, no_linear=None)
        self.cbn2 = CBN2d(in_channels, out_channels, kernel_size=1, padding=0,
                          no_linear=nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))

    def forward(self, inputs):
        outputs1 = self.cbn1(inputs) # 600 -> 300 -> 150 -> 75 -> 38
        outputs2 = self.cbn2(inputs)
        
        return F.relu(outputs1 + outputs2)


class AttentionFeaturePyramidFusion(nn.Module):

    def __init__(self, lower_channels, higher_channels, out_channels, stride, output_padding):
        super(AttentionFeaturePyramidFusion, self).__init__()

        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(higher_channels, out_channels, 3, stride, 1, output_padding),
            CBN2d(out_channels, out_channels)
        )
        self.cbn = CBN2d(lower_channels, out_channels)
        self.attn = nn.Conv2d(2*out_channels, 2, 3, 1, 1)

    def forward(self, inputs_lower, inputs_higher):

        outputs_higher = self.up_sample(inputs_higher)
        outputs_lower = self.cbn(inputs_lower)

        attn_weight = torch.softmax(self.attn(torch.cat([outputs_higher, outputs_lower], dim=1)), dim=1)
        outputs = outputs_higher * attn_weight[:, 0:1, :, :] + outputs_lower * attn_weight[:, 1:, :, :]
        
        return outputs


""" 
# attention with sigmoid
class AttentionFeaturePyramidFusion(nn.Module):

    def __init__(self, lower_channels, higher_channels, out_channels, stride, output_padding):
        super(AttentionFeaturePyramidFusion, self).__init__()

        self.up_sample = nn.Sequential(
            nn.ConvTranspose2d(higher_channels, out_channels, 3, stride, 1, output_padding),
            CBN2d(out_channels, out_channels)
        )
        self.cbn = CBN2d(lower_channels, out_channels)
        self.attn = nn.Conv2d(2*out_channels, 1, 3, 1, 1)

    def forward(self, inputs_lower, inputs_higher):

        outputs_higher = self.up_sample(inputs_higher)
        outputs_lower = self.cbn(inputs_lower)

        attn_weight = torch.sigmoid(self.attn(torch.cat([outputs_higher, outputs_lower], dim=1)))
        outputs = outputs_higher * (1 - attn_weight) + outputs_lower * attn_weight
        
        return outputs
"""


class CPGFCN(nn.Module):

    def __init__(self, 
                 in_channels=64, 
                 encoder_channels=[32, 64, 128, 128], 
                 stride=[2, 2, 2, 1],
                 decoder_channels=[96, 64, 64], 
                 output_padding=[1, 1, 1]):
        super(CPGFCN, self).__init__()
        assert len(decoder_channels) == len(output_padding) and len(encoder_channels) == len(stride)
        assert len(decoder_channels) <= len(encoder_channels)

        self.encoder = nn.ModuleList()
        for in_c, out_c, s in zip(([in_channels] + encoder_channels)[:-1], encoder_channels, stride):
            self.encoder.append(DualDownSamplingBlock(in_c, out_c, s))
        
        self.decoder = nn.ModuleList()
        for lower_c, higher_c, out_c, s, op in zip(([in_channels]+encoder_channels)[:len(decoder_channels)][::-1],
                                                   encoder_channels[-2:-1]+decoder_channels[:-1],
                                                   decoder_channels, stride[:len(decoder_channels)][::-1], output_padding):
            self.decoder.append(AttentionFeaturePyramidFusion(lower_c, higher_c, out_c, s, op)) 

    def forward(self, inputs):
        """
        Param:
            inputs: with shape of :math:`(N,C,H,W)`, where N is batch size
        """
        encoder_outputs = []
        outputs = inputs
        for layer in self.encoder:
            outputs = layer(outputs)
            encoder_outputs.append(outputs)
        
        outputs = encoder_outputs[-1]
        for layer, inputs_lower in zip(self.decoder, ([inputs]+encoder_outputs)[:len(self.decoder)][::-1]):
            outputs = layer(inputs_lower, outputs)
        
        return outputs

   
class ResFCN(nn.Module):
    
    def __init__(self, in_channels=7, out_channels=32, encoder_channels=[64, 64, 256, 1024, 2048, 3072, 4096]):
        super(ResFCN, self).__init__()

        self.in_channels = in_channels
        cs = encoder_channels

        self.pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))           
        )
        self.pool5 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))           
        )
        self.pool7 = nn.Sequential(
                nn.MaxPool2d(kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))           
        )

        self.conv1 = (nn.Conv2d(4 * in_channels, cs[0], kernel_size=3, stride=1,padding=1))
        self.bn1 = nn.BatchNorm2d(cs[0], momentum=0.9, eps=1e-5)

        self.down1 = (nn.Conv2d(cs[0], cs[1], kernel_size=3, stride=2,padding=1))
        self.bn2 = nn.BatchNorm2d(cs[1], momentum=0.9, eps=1e-5)

        self.down2 = (nn.Conv2d(cs[1], cs[2], kernel_size=3, stride=2,padding=1))
        self.bn3 = nn.BatchNorm2d(cs[2], momentum=0.9, eps=1e-5)

        self.down3 = (nn.Conv2d(cs[2], cs[3], kernel_size=3, stride=2,padding=1))
        self.bn4 = nn.BatchNorm2d(cs[3], momentum=0.9, eps=1e-5)

        self.res1 = (nn.Conv2d(cs[3], cs[3], kernel_size=3, stride=1,padding=1))
        self.bn5 = nn.BatchNorm2d(cs[3], momentum=0.9, eps=1e-5)

        self.res2 = (nn.Conv2d(cs[4], cs[3], kernel_size=3, stride=1,padding=1))
        self.bn6 = nn.BatchNorm2d(cs[3], momentum=0.9, eps=1e-5)

        self.res3 = (nn.Conv2d(cs[5], cs[3], kernel_size=3, stride=1,padding=1))
        self.bn7 = nn.BatchNorm2d(cs[3], momentum=0.9, eps=1e-5)

        self.conv2 = (nn.Conv2d(cs[6], cs[3], kernel_size=3, stride=1,padding=1))

        self.deconv1 = (nn.ConvTranspose2d(cs[3], cs[2], kernel_size=3, stride=2,padding=1,output_padding=1))
        self.bn8 = nn.BatchNorm2d(cs[2], momentum=0.9, eps=1e-5)

        self.deconv2 = (nn.ConvTranspose2d(cs[2], cs[1], kernel_size=3, stride=2,padding=1,output_padding=1))
        self.bn9 = nn.BatchNorm2d(cs[1], momentum=0.9, eps=1e-5)

        self.deconv3 = (nn.ConvTranspose2d(cs[1], cs[0], kernel_size=3, stride=2,padding=1,output_padding=1))
        self.bn10 = nn.BatchNorm2d(cs[0], momentum=0.9, eps=1e-5)

        self.conv3 = (nn.Conv2d(cs[0], out_channels, kernel_size=3, stride=1,padding=1))

    def forward(self, x):
        leaky_relu = functools.partial(F.leaky_relu, negative_slope=0.2)

        x3 = self.pool3(x)
        x5 = self.pool5(x)
        x7 = self.pool7(x)
        x = torch.cat((x, x3, x5, x7),1)

        x1_0 = leaky_relu(self.bn1(self.conv1(x)))  # 4*16*64*2048
        x1_1 = leaky_relu(self.bn2(self.down1(x1_0)))  # 4*64*32*1024
        x1_2 = leaky_relu(self.bn3(self.down2(x1_1)))  # 4*256*16*512
        x1_3 = leaky_relu(self.bn4(self.down3(x1_2)))  # 4*1024*8*256

        r1 = leaky_relu(self.bn5(self.res1(x1_3)))  # 4*1024*8*256
        r1 = torch.cat((r1, x1_3),1)  # 4*2048*8*256
        r2 = leaky_relu(self.bn6(self.res2(r1)))  # 4*1024*8*256
        r2 = torch.cat((r2, r1),1)  # 4*3072*8*256
        r3 = leaky_relu(self.bn7(self.res3(r2)))  # 4*1024*8*256
        r3 = torch.cat((r3, r2),1)  # 4*4096*8*256
        
        x = leaky_relu((self.conv2(r3))) + x1_3  # 4*1024*8*256
        x = leaky_relu(self.bn8(self.deconv1(x))) + x1_2  # 4*256*16*512
        x = leaky_relu(self.bn9(self.deconv2(x))) + x1_1  # 4*64*32*1024
        x = leaky_relu(self.bn10(self.deconv3(x))) + x1_0  # 4*16*64*2048
        x = leaky_relu((self.conv3(x)))  # 4*32*64*2048
        return x
