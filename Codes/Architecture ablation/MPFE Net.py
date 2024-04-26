import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchvision import models as resnet_model

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)

        return x
class DepthwiseSeparableConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=3, groups=in_channels,padding=1)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    

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
        self.conv1x1_1 = conv1x1(128,256)
        self.conv1x1_2 = conv1x1(256,512)


        self.dwconv1 = DepthwiseSeparableConv(512,512)
        self.dwconv2 = DepthwiseSeparableConv(256,256)
        self.dwconv3 = DepthwiseSeparableConv(128,128)
        # self.dwconv4 = DepthwiseSeparableConv(512,32)

        self.dconv1 = DoubleConv(512,256)
        self.dconv2 = DoubleConv(512,128)
        self.dconv3 = DoubleConv(512,64)


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


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.layer = nn.LayerNorm(dim)
        self.gap1 = nn.AdaptiveAvgPool1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gap2 = nn.AdaptiveAvgPool1d(dim)


    def forward(self, x):
        x_matrix = rearrange(x, 'b n (h w) -> b n h w', h=14, w=14)
        x_transpose = torch.transpose(x_matrix, -1, -2)
        x_transpose_vector = rearrange(x_transpose, 'b n h w -> b n (h w)')
        x_gap_1 = self.gap1(x_transpose_vector)
        x_gap_1_dropout = self.dropout(x_gap_1)
        x_gap_2 = self.gap2(x_gap_1_dropout)
        x_gap_2_matrix = rearrange(x_gap_2, 'b n (h w) -> b n h w', h=14, w=14)

        x_gap_2 = x_gap_2_matrix + x_transpose
        x_gap_2_dropout = self.dropout(x_gap_2)
        output = torch.transpose(x_gap_2_dropout, -1, -2)
        output = rearrange(output, 'b n h w -> b n (h w)')
        
        return output + x


class Conv3x3_2(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, dilation=2):
        super(Conv3x3_2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=groups,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv3x3_3(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, dilation=3):
        super(Conv3x3_3, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, groups=groups,
                              dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
            # 主干网络
                # 第一阶段
            self.avg1 = nn.AvgPool2d(2, 2)
            self.max1 = nn.MaxPool2d(2, 2)
            self.con1 = nn.Conv2d(65, 130, 3, stride=2,padding=1, bias=False)
            self.conv1 = nn.Conv2d(260, 65, 3, 1, 1, bias=False)

                # 第二阶段
            self.gmp = nn.AdaptiveMaxPool1d(1)
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(1, 196)
            self.conv = nn.Conv1d(195, 65, 1, bias=False)


            # 左边分支
            self.fc_1 = nn.Linear(196, 392)
            self.fc_2 = nn.Linear(1, 196)

            # 右边分支
            self.conv3x3_2_1 = Conv3x3_2(65, 130)
            self.conv3x3_2_2 = Conv3x3_2(130, 260)
            self.conv3x3_3_1 = Conv3x3_3(65, 130)
            self.conv3x3_3_2 = Conv3x3_3(130, 260)
            self.dws = DepthwiseSeparableConv1D(455, 65)

    def forward(self, x):
        for attn, ff in self.layers:

            # 主干网络
                # 第一阶段
            x_matrix = x.view(4, 65, 14, 14)
            x_avg1 = self.avg1(x_matrix)
            x_max1 = self.max1(x_matrix)
            x_con1 = self.con1(x_matrix)
            x_1 = torch.cat((x_avg1, x_max1, x_con1), dim=1)
            x_1_conv1 = self.conv1(x_1)
            x_1_matrix = F.interpolate(x_1_conv1, size=(14, 14), mode='bilinear', align_corners=True)
            x_1_output_1 = rearrange(x_1_matrix, "b h n d -> b h (n d)")

                # 第二阶段
            
            x_gap = self.gap(x_1_output_1)
            x_gmp = self.gmp(x_1_output_1)
            x_gap_gmp = x_gmp + x_gap
            x1 = self.fc(x_gap)
            x2 = self.fc(x_gap_gmp)
            x3 = self.fc(x_gmp)
            x_fc = torch.cat([x1, x2, x3], dim=1)
            x_fc_output = self.conv(x_fc)
            x_1_output = x_fc_output + x_1_output_1


            # 左边分支
            x_attn = attn(x)
            x_attn_matrix = x_attn.view(4, 65, 14, 14)
            x_attn_matrix_transpose = torch.transpose(x_attn_matrix, -1, -2)
            x_attn_vector_transpose = rearrange(x_attn_matrix_transpose, 'b h n d -> b h (n d)')
            x_fc_1 = self.fc_1(x_attn_vector_transpose)
            x_avg = nn.AdaptiveAvgPool1d(1)(x_fc_1)
            x_fc_2 = self.fc_2(x_avg)
            x_fc_2 = x_fc_2 + x_attn_vector_transpose
            x_fc_2_matrix = rearrange(x_fc_2, "b h (n d) -> b h n d", n=14, d=14)
            x_2_output_matrix = torch.transpose(x_fc_2_matrix, -1, -2)
            x_2_output = rearrange(x_2_output_matrix, "b h n d -> b h (n d)")

            # 右边分支
            x_2_1 = self.conv3x3_2_1(x_attn_matrix)
            x_3_1 = self.conv3x3_3_1(x_attn_matrix)
            x_2_2 = self.conv3x3_2_2(x_2_1 + x_3_1)
            x_3_2 = self.conv3x3_3_2(x_3_1 + x_2_1)
            x_3_output_matrix = x_3_2 + x_2_2
            x_3_output = rearrange(x_3_output_matrix, "b h n d -> b h (n d)")

            x_pre_output = torch.cat((x_1_output, x_2_output, x_3_output, x), dim=1)

            x_output = self.dws(x_pre_output)

            x = ff(x_output) + x_output

        return x


class encoder(nn.Module):
    # def __init__(self):
    #     super(encoder,self).__init__()

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.channelTrans = nn.Conv2d(in_channels=65, out_channels=512, kernel_size=1, padding=0)

    def forward(self, x):
        x_vit = x
        x_vit = self.to_patch_embedding(x_vit)
        b, n, _ = x_vit.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x_vit = torch.cat((cls_tokens, x_vit), dim=1)
        x_vit += self.pos_embedding[:, :(n + 1)]
        x_vit = self.dropout(x_vit)

        vit_layerInfo = []
        for i in range(4):  # 设置深度的地方[6, 64+1, dim=196]
            x_vit = self.transformer(x_vit)
            vit_layerInfo.append(x_vit)

        return vit_layerInfo



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


class Conv(nn.Module):
    # 初始化
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        # 这里执行的是双卷积的初始化操作，用Sequential这个序列把他保存起来
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),  # 第一次卷积，卷积核3*3
            nn.BatchNorm2d(out_channels),  # 批标准化，用于放置梯度爆炸和梯度消失
            nn.ReLU(inplace=True),  # 激活函数，其中inplace参数用于检查是否进行覆盖运算

        )

    # 按序，将初始化的组件，进行拼装。这里不需要拼装，因为初始化的过程中已经把顺序定好了
    def forward(self, x):
        return self.conv(x)


class Conv_32(nn.Module):
    # 初始化
    def __init__(self, in_channels, out_channels):
        super(Conv_32, self).__init__()
        # 这里执行的是双卷积的初始化操作，用Sequential这个序列把他保存起来
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),  # 第一次卷积，卷积核3*3
            nn.BatchNorm2d(out_channels),  # 批标准化，用于放置梯度爆炸和梯度消失
            nn.ReLU(inplace=True),  # 激活函数，其中inplace参数用于检查是否进行覆盖运算

        )

    # 按序，将初始化的组件，进行拼装。这里不需要拼装，因为初始化的过程中已经把顺序定好了
    def forward(self, x):
        return self.conv(x)


class UPConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UPConv, self).__init__()
        self.conv = nn.Sequential(

        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ImageSegmentationNet(nn.Module):
    def __init__(self):
        super(ImageSegmentationNet, self).__init__()

    def forward(self, x):
        # 假设 x 的尺寸为 (B, C, H, W)
        B, C, H, W = x.size()

        # 确保图像的高和宽可以被2整除
        assert H % 2 == 0 and W % 2 == 0, '图像的高和宽必须能被2整除。'

        # 分割图像
        upper_left = x[:, :, :H // 2, :W // 2]
        upper_right = x[:, :, :H // 2, W // 2:]
        lower_left = x[:, :, H // 2:, :W // 2]
        lower_right = x[:, :, H // 2:, W // 2:]

        # 可以选择根据需要对这些分割后的图像进行进一步的操作
        return upper_left, upper_right, lower_left, lower_right


class ImageSegmentationNet1(nn.Module):
    def __init__(self):
        super(ImageSegmentationNet1, self).__init__()

    def forward(self, input_tensor):
        # 假设 x 的尺寸为 (B, C, H, W)
        _, _, height, width = input_tensor.size()

        # 计算开始点
        start_x = (width - 112) // 2
        start_y = (height - 112) // 2

        # 裁剪图像
        cropped = input_tensor[:, :, start_y:start_y + 112, start_x:start_x + 112]

        return cropped


class ZeroPadToSize(nn.Module):
    def __init__(self, target_height=224, target_width=224):
        """
        初始化一个用于将输入张量通过周围填充0扩展到指定大小的模块。

        参数:
        target_height - 目标高度
        target_width - 目标宽度
        """
        super(ZeroPadToSize, self).__init__()
        self.target_height = target_height
        self.target_width = target_width

    def forward(self, input_tensor):
        """
        对输入张量进行处理。

        参数:
        input_tensor - 输入的张量，形状为 (B, C, H, W)

        返回:
        填充后的张量，其大小为 (B, C, target_height, target_width)
        """
        _, _, height, width = input_tensor.size()

        # 计算需要在每个方向上填充的大小
        pad_height = (self.target_height - height) // 2
        pad_width = (self.target_width - width) // 2

        # 对于奇数差异，额外填充应加在右侧或下侧
        extra_pad_height = (self.target_height - height) % 2
        extra_pad_width = (self.target_width - width) % 2

        # 应用填充
        padded_tensor = F.pad(input_tensor,
                              (pad_width, pad_width + extra_pad_width,
                               pad_height, pad_height + extra_pad_height),
                              'constant', 0)
        return padded_tensor


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder = encoder(image_size=224, patch_size=28, num_classes=2, dim=196, depth=6, heads=16, mlp_dim=2048)

        self.DoubleConv3_64 = DoubleConv(3, 64)
        self.DoubleConv16_64 = DoubleConv(16, 64)
        self.DoubleConv8_16 = DoubleConv(8, 16)
        self.DoubleConv16_32 = DoubleConv(16, 32)
        self.DoubleConv16_16 = DoubleConv(16, 16)
        self.DoubleConv32_64 = DoubleConv(32, 64)
        self.DoubleConv64_64 = DoubleConv(64, 64)
        self.DoubleConv32_32 = DoubleConv(32, 32)
        self.DoubleConv64_128 = DoubleConv(64, 128)
        self.DoubleConv128_128 = DoubleConv(128, 128)
        self.DoubleConv128_256 = DoubleConv(128, 256)
        self.DoubleConv256_256 = DoubleConv(256, 256)
        self.DoubleConv256_512 = DoubleConv(256, 512)
        self.DoubleConv512_512 = DoubleConv(512, 512)
        self.DoubleConv512_1024 = DoubleConv(512, 1024)
        self.DoubleConv1024_2048 = DoubleConv(1024, 2048)
        self.DoubleConv1024_1024 = DoubleConv(1024, 1024)
        self.DoubleConv = DoubleConv(64, 128)
        self.up64_32 = UPConv(64, 32)
        self.up32_16 = UPConv(32, 16)
        self.up3_64 = UPConv(3,64)
        self.up128_64 = UPConv(128, 64)
        self.up16_8 = UPConv(16, 8)
        self.up512_256 = UPConv(512, 256)
        self.up1024_512 = UPConv(1024, 512)
        self.up256_128 = UPConv(256, 128)
        self.up64_64 = UPConv(64, 64)
        self.conv1024_512 = Conv(1024, 512)
        self.conv128 = Conv(65 + 128, 128)
        self.conv256 = Conv(65 + 256, 256)
        self.conv512 = Conv(512 + 65, 512)
        self.conv384_512 = Conv(128 + 256, 512)
        self.conv64 = Conv(64, 3)
        self.conv512_256 = Conv(512, 256)
        self.conv128_64 = Conv(128, 64)
        self.conv256_128 = Conv(256, 128)

    def forward(self, x,y):

        x_1 = self.up3_64(x)  # 64*224*224

        x_2 = self.pool(x_1)  # 112*112*64
        x_3 = self.DoubleConv64_128(x_2)  # 112*112*128
        x_3 = self.pool(x_3)  # 56*56*128
        x_y= torch.cat((x_3, y), dim=1)
        x_y=self.conv128(x_y) # 65+128----128
        x_4 = self.DoubleConv128_256(x_y)  # 56*56*256
        x_4 = self.pool(x_4)  # 28*28*256
        x_5 = self.DoubleConv256_512(x_4)  # 28*28*512
        x_5 = self.pool(x_5)  # 14*14*512

        return x_5


class NET1(nn.Module):
    def __init__(self):
        super(NET1, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder = encoder(image_size=224, patch_size=28, num_classes=2, dim=196, depth=6, heads=16, mlp_dim=2048)

        self.DoubleConv3_64 = DoubleConv(3, 64)
        self.DoubleConv16_64 = DoubleConv(16, 64)
        self.DoubleConv8_16 = DoubleConv(8, 16)
        self.DoubleConv16_32 = DoubleConv(16, 32)
        self.DoubleConv16_16 = DoubleConv(16, 16)
        self.DoubleConv32_64 = DoubleConv(32, 64)
        self.DoubleConv64_64 = DoubleConv(64, 64)
        self.DoubleConv32_32 = DoubleConv(32, 32)
        self.DoubleConv64_128 = DoubleConv(64, 128)
        self.DoubleConv128_128 = DoubleConv(128, 128)
        self.DoubleConv128_256 = DoubleConv(128, 256)
        self.DoubleConv256_256 = DoubleConv(256, 256)
        self.DoubleConv256_512 = DoubleConv(256, 512)
        self.DoubleConv512_512 = DoubleConv(512, 512)
        self.DoubleConv512_1024 = DoubleConv(512, 1024)
        self.DoubleConv1024_2048 = DoubleConv(1024, 2048)
        self.DoubleConv1024_1024 = DoubleConv(1024, 1024)
        self.DoubleConv = DoubleConv(64, 128)
        self.up64_32 = UPConv(64, 32)
        self.up32_16 = UPConv(32, 16)
        self.up64_3 = UPConv(64, 3)
        self.up128_64 = UPConv(128, 64)
        self.up16_8 = UPConv(16, 8)
        self.up512_256 = UPConv(512, 256)
        self.up1024_512 = UPConv(1024, 512)
        self.up256_128 = UPConv(256, 128)
        self.up64_64 = UPConv(64, 64)
        self.conv1024_512 = Conv(1024, 512)
        self.conv128 = Conv(65 + 128+64, 128)
        self.conv256 = Conv(65 + 256+128, 256)
        self.conv512 = Conv(512 + 65, 512)
        self.conv384_512 = Conv(128 + 256, 512)
        self.conv64 = Conv(64, 3)
        self.conv512_256 = Conv(512, 256)
        self.conv128_64 = Conv(128, 64)
        self.conv256_128 = Conv(256, 128)

        self.sge = ImageSegmentationNet()
        self.sge1 = ImageSegmentationNet1()
        self.pad = ZeroPadToSize()

        self.vitLayer_UpConv = nn.ConvTranspose2d(65, 65, kernel_size=2, stride=2)

    # def forward(self, x,y):
    #
    #     x_2 = self.up512_256(x)
    #     x_2 = self.up256_128(x_2)
    #
    #     x_y= torch.cat((x_2, y), dim=1)
    #     x_y=self.conv128(x_y) # 56+256----256*56*56
    #
    #     return x_y
    def forward(self, x, y, z,a,b):
        x_2 = self.up512_256(x)  # 28
        x_z = self.conv256(torch.cat((x_2, z,a), dim=1))
        x_2 = self.up256_128(x_z)  # 56

        x_y = torch.cat((x_2, y,b), dim=1)
        x_y = self.conv128(x_y)  # 56+256----256*56*56

        return x_y


class MergeFourImages(nn.Module):
    def __init__(self):
        """
        初始化一个用于将四个图像部分拼接回原图的模块。
        """
        super(MergeFourImages, self).__init__()

    def forward(self, top_left, top_right, bottom_left, bottom_right):

        # 沿宽度方向拼接上半部分
        top_half = torch.cat([top_left, top_right], dim=3)
        # 沿宽度方向拼接下半部分
        bottom_half = torch.cat([bottom_left, bottom_right], dim=3)

        # 沿高度方向拼接上半部分和下半部分
        merged_tensor = torch.cat([top_half, bottom_half], dim=2)

        return merged_tensor


class ENET(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], out_channels=1):
        super(ENET, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder = encoder(image_size=224, patch_size=28, num_classes=2, dim=196, depth=6, heads=16, mlp_dim=2048)
        self.finalconv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.DoubleConv3_64 = DoubleConv(3, 64)
        self.DoubleConv16_64 = DoubleConv(16, 64)
        self.DoubleConv8_16 = DoubleConv(8, 16)
        self.DoubleConv16_32 = DoubleConv(16, 32)
        self.DoubleConv16_16 = DoubleConv(16, 16)
        self.DoubleConv32_64 = DoubleConv(32, 64)
        self.DoubleConv64_64 = DoubleConv(64, 64)
        self.DoubleConv32_32 = DoubleConv(32, 32)
        self.DoubleConv64_128 = DoubleConv(64, 128)
        self.DoubleConv128_128 = DoubleConv(128, 128)
        self.DoubleConv128_256 = DoubleConv(128, 256)
        self.DoubleConv256_256 = DoubleConv(256, 256)
        self.DoubleConv256_512 = DoubleConv(256, 512)
        self.DoubleConv512_512 = DoubleConv(512, 512)
        self.DoubleConv512_1024 = DoubleConv(512, 1024)
        self.DoubleConv1024_2048 = DoubleConv(1024, 2048)
        self.DoubleConv1024_1024 = DoubleConv(1024, 1024)
        self.DoubleConv = DoubleConv(64, 128)
        self.up64_32 = UPConv(64, 32)
        self.up32_16 = UPConv(32, 16)
        self.up64_3 = UPConv(64, 3)
        self.up128_64 = UPConv(128, 64)
        self.up16_8 = UPConv(16, 8)
        self.up512_256 = UPConv(512, 256)
        self.up1024_512 = UPConv(1024, 512)
        self.up256_128 = UPConv(256, 128)
        self.up64_64 = UPConv(64, 64)
        self.up512_512 = UPConv(512,512)
        self.final_conv = Conv_32(64, out_channels)
        self.conv1024_512 = Conv(1024, 512)
        self.conv128 = Conv(65 + 128, 128)
        self.conv256 = Conv(65 + 256, 256)
        self.conv512 = Conv(512 + 65, 512)
        self.conv384_512 = Conv(128 + 256, 512)
        self.conv64 = Conv(64,3)
        self.conv512_256 = Conv(512, 256)
        self.conv128_64 = Conv(128, 64)
        self.conv256_128 = Conv(256, 128)

        self.sge = ImageSegmentationNet()
        self.sge1 = ImageSegmentationNet1()
        self.pad = ZeroPadToSize()
        self.NET=NET()
        self.NET1 = NET1()
        self.meg=MergeFourImages()

        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.vitLayer_UpConv = nn.ConvTranspose2d(65, 65, kernel_size=2, stride=2)
        self.bottelneck = BottleNeck()

    def forward(self, x):
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)
        e1 = self.encoder1(e0)  # torch.Size([4, 64,112
        e2 = self.encoder2(e1)  # torch.Size([4, 128,56
        e3 = self.encoder3(e2)  # torch.Size([4, 256,28
        e4 = self.encoder4(e3)  # torch.Size([4, 512, 14
        xe4=self.up512_512(e4)

        x_middle = self.sge1(x)
        # 中间强化
        x_middle64 = self.DoubleConv3_64(x_middle)
        x_middle = self.up64_64(x_middle64)
        x_middle_f = self.pool(x_middle)
        x_middle_f = self.pad(x_middle_f)
        x_middle_ff = self.conv64(x_middle_f)  # 3,224,224
        # 强化还原,准备进入vit
        x_vp = self.DoubleConv3_64(x) + x_middle_f  # 64，224，224
        x_vp=self.conv64(x_vp)  # 3,224,224
        # 四周强化
        # vit_layerInfo_f = self.encoder(x_middle_ff)
        # vit_layerInfo_f = vit_layerInfo_f[::-1]
        #
        # v4 = vit_layerInfo_f[3].view(4, 65, 14, 14)  # 224
        # v4_1 = self.vitLayer_UpConv(v4)
        # v4_3 = self.vitLayer_UpConv(v4_1)
        # v4_f = self.vitLayer_UpConv(v4_3)

        vit_layerInfo = self.encoder(x_vp)
        vit_layerInfo = vit_layerInfo[::-1]  # 翻转，呈正序。0表示第四层...3表示第一层

        v1 = vit_layerInfo[1].view(4, 65, 14, 14)  # 28
        v1_1 = self.vitLayer_UpConv(v1)

        v2 = vit_layerInfo[2].view(4, 65, 14, 14)  # 56
        v2_1 = self.vitLayer_UpConv(v2)
        v2_2 = self.vitLayer_UpConv(v2_1)

        v4 = vit_layerInfo[3].view(4, 65, 14, 14)  # 112
        v4_1 = self.vitLayer_UpConv(v4)
        v4_2 = self.vitLayer_UpConv(v4_1)
        v4_4 = self.vitLayer_UpConv(v4_2)
        # 第二分支开始，辅助分枝暂时结束


        # xe1 = self.DoubleConv3_64(x)
        # xe1_p = self.pool(xe1)# 112
        # # xe2_p1= torch.cat((v3_3, e1), dim=1)  # 通道数为65+64
        # # xe2_p2=self.conv64(xe2_p1)+xe1_p
        # xe2 = self.DoubleConv64_128(xe1_p)
        # xe2_p = self.pool(xe2)# 56
        # #
        # # xe3_p1 = torch.cat((v2_2, e2), dim=1)  # 通道数为65+128
        # # xe3_p2 = self.conv128(xe3_p1) + xe2_p
        # xe3 = self.DoubleConv128_256(xe2_p)
        # xe3_p = self.pool(xe3)#28
        #
        # # xe4_p1 = torch.cat((v1_1, e3), dim=1)  # 通道数为65+256
        # # xe4_p2 = self.conv256(xe4_p1) + xe3_p
        # xe4 = self.DoubleConv256_512(e3)
        # xe4_p = self.pool(xe4)

        upper_left, upper_right, lower_left, lower_right = self.sge(xe4)  # 14
        upper_leftv, upper_rightv, lower_leftv, lower_rightv = self.sge(v1_1)  # btn
        x_u11 = self.conv512(torch.cat((upper_left, upper_leftv), dim=1))
        x_u22 = self.conv512(torch.cat((upper_right, upper_rightv), dim=1))
        x_l11 = self.conv512(torch.cat((lower_left, lower_leftv), dim=1))
        x_l22 = self.conv512(torch.cat((lower_right, lower_rightv), dim=1))

        x_ed=self.meg(x_u11, x_u22, x_l11, x_l22)  # 28
        x_ed1= self.bottelneck(x_ed, xe4)
        x_u11 , x_u22, x_l11 ,  x_l22= self.sge(x_ed1)

        # x_u1=self.NET(upper_left,upper_leftv)
        # x_u2 = self.NET(upper_right, upper_rightv)
        # x_l1 = self.NET(lower_left, lower_leftv)
        # x_l2 = self.NET(lower_right, lower_rightv)
        a,b,c,d = self.sge(e2)
        aa,bb,cc,dd = self.sge(e1)

        upper_leftvv, upper_rightvv, lower_leftvv, lower_rightvv = self.sge(v4_4)  #28

        upper_leftv, upper_rightv, lower_leftv, lower_rightv = self.sge(v2_2)  #56
        x_ue1 = self.NET1(x_u11, upper_leftvv, upper_leftv,a,aa)
        x_ue2 = self.NET1(x_u22, upper_rightvv, upper_rightv,b,bb)
        x_le1 = self.NET1(x_l11, lower_leftvv, lower_leftv,c,cc)
        x_le2 = self.NET1(x_l22, lower_rightvv, lower_rightv,d,dd)

        x_ff = self.conv128_64(self.meg(x_ue1, x_ue2, x_le1, x_le2))
        x_f = self.up64_64(x_ff)
        out = self.final_conv(x_f)
        return out

if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)
    model = ENET()
    preds = model(x)
    print(x.shape)
    print(preds.shape)
