import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from thop import profile


class DoubleConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1, self).__init__()
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


class DepthwiseSeparableConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv1, self).__init__()
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
        self.conv1x1_1 = conv1x1(128,256)
        self.conv1x1_2 = conv1x1(256,512)

        self.dwconv1 = DepthwiseSeparableConv1(512,512)
        self.dwconv2 = DepthwiseSeparableConv1(256,256)
        self.dwconv3 = DepthwiseSeparableConv1(128,128)
        # self.dwconv4 = DepthwiseSeparableConv(512,32)

        self.dconv1 = DoubleConv1(512,256)
        self.dconv2 = DoubleConv1(512,128)
        self.dconv3 = DoubleConv1(512,64)


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


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(  # nn.Sequential 有序容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace 是否进行覆盖运算
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(
            self, in_channels, out_channels, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()  # 将多个Module加入list，但不存在实质性顺序，参考python的list
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck1 = DoubleConv(features[-1], features[-1]*2)
        self.bottleneck=BottleNeck()
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # decoder part
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, x)

        x = self.bottleneck1(x)


        skip_connections = skip_connections[::-1]

        # encoder part
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


# x = torch.randn(4, 3, 224,224)
# model = UNET(3,1)
# preds = model(x)
# print(x.shape)
# print(preds.shape)
