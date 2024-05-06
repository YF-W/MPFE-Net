import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import models as resnet_model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels,padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv1x1,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,1)

    def forward(self,x):
        x = self.conv(x)

        return x

class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck,self).__init__()
        self.conv1x1_1 = conv1x1(64,128)
        self.conv1x1_2 = conv1x1(128,256)


        self.dwconv1 = DepthwiseSeparableConv(256,256)
        self.dwconv2 = DepthwiseSeparableConv(128,128)
        self.dwconv3 = DepthwiseSeparableConv(64,64)
        # self.dwconv4 = DepthwiseSeparableConv(512,32)

        self.dconv1 = DoubleConv(256,128)
        self.dconv2 = DoubleConv(256,64)
        self.dconv3 = DoubleConv(256,32)


    def forward(self,x,y):
        height, width = x.size()[2:]
        v1 = self.dconv1(x)
        v2 = self.dconv2(x)
        v3 = self.dconv3(x)
        # v4 = self.conv1x1_4(x)

        e1 = self.dconv1(y)
        e2 = self.dconv2(y)
        e3 = self.dconv3(y)
        # e4 = self.dwconv4(y)

        f1 = torch.cat([v1,e1],dim=1)
        f2 = torch.cat([v2,e2],dim=1)
        f3 = torch.cat([v3,e3],dim=1)
        # f4 = torch.cat([v4,e4],dim=1)


        f3_1 = self.dwconv3(f3)
        f2_1 = self.dwconv2(f2)
        f1_1 = self.dwconv1(f1)
        f3_2 = self.conv1x1_1(f3_1)
        f2_2 = f2_1 + f3_2
        f2_2_result1 = F.adaptive_avg_pool2d(f2_2, (height, 1))
        f2_2_result2 = F.adaptive_avg_pool2d(f2_2, (1,width))
        f2_2_result = torch.matmul(f2_2_result1,f2_2_result2)
        f2_3 = self.conv1x1_2(f2_2_result)
        f1_2 = f1_1*f2_3
        f1_3= f1_2+x+y

        return f1_3
class batchnorm_relu(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.bn(inputs)
        x = self.relu(x)
        return x


class residual_block(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        """ Convolutional layer """
        self.b1 = batchnorm_relu(in_c)
        self.c1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride)
        self.b2 = batchnorm_relu(out_c)
        self.c2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        """ Shortcut Connection (Identity Mapping) """
        self.s = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, stride=stride)

    def forward(self, inputs):
        x = self.b1(inputs)
        x = self.c1(x)
        x = self.b2(x)
        x = self.c2(x)
        s = self.s(inputs)

        skip = x + s
        return skip


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r = residual_block(in_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat((x, skip), dim=1)
        x = self.r(x)
        return x


class build_resunet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder 1 """
        self.c11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.br1 = batchnorm_relu(64)
        self.c12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c13 = nn.Conv2d(3, 64, kernel_size=1, padding=0)

        """ Encoder 2 and 3 """
        self.r2 = residual_block(64, 128, stride=2)
        self.r3 = residual_block(128, 256, stride=2)

        """ Bridge """
        self.r4 = residual_block(256, 512, stride=2)

        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256, 128)
        self.d3 = decoder_block(128, 64)

        """ Output """
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        self.sigmoid = torch.sigmoid
        self.bottleneck=BottleNeck()
        self.conv =DoubleConv(256,512)
        self.pool = nn.AvgPool2d(2,2)

    def forward(self, inputs):
        """ Encoder 1 """
        x = self.c11(inputs)
        x = self.br1(x)
        x = self.c12(x)
        s = self.c13(inputs)
        skip1 = x + s

        """ Encoder 2 and 3 """
        skip2 = self.r2(skip1)
        skip3 = self.r3(skip2)

        """ Bridge """
        # b = self.r4(skip3)
        b=self.bottleneck(skip3,skip3)
        b=self.conv(b)
        b=self.pool(b)


        """ Decoder """
        d1 = self.d1(b, skip3)
        d2 = self.d2(d1, skip2)
        d3 = self.d3(d2, skip1)

        """ output """
        output = self.output(d3)
        # output = self.sigmoid(output)

        return output


def test():
    x = torch.randn((4, 3, 224, 224))
    model = build_resunet()
    preds = model(x)
    print(preds.shape)
    print(x.shape)


if __name__ == "__main__":
    test()
