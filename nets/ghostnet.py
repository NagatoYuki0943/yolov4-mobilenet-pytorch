import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ghost_net']

def _make_divisible(v, divisor, min_value=None):
    """
    将卷积核个数(输出通道个数)调整为最接近round_nearest的整数倍,就是8的整数倍,对硬件更加友好
    v:          输出通道个数
    divisor:    奇数,必须将ch调整为它的整数倍
    min_value:  最小通道数
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#--------------------------------------------#
#   sigmoid(x)   = \frac 1 {1 + e^{-x}}
#   h-sigmoid(x) = \frac {ReLU6(x + 3)} {6}
#--------------------------------------------#
def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)

        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)

        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size, stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

#------------------------------------------------------#
#   1x1Conv降低通道数,特征浓缩
#   3x3DWConv对降低的通道数进行计算
#   1x1Conv和3x3DWConv的输出拼接返回
#------------------------------------------------------#
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup

        # 输出通道数 / 2
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        #------------------------------------------------------#
        #   1x1Conv降低通道数,特征浓缩 通道数变为 out_channel/2
        #   跨通道的特征提取
        #------------------------------------------------------#
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        #------------------------------------------------------#
        #   3x3DWConv对降低的通道数进行计算
        #   跨特征点的特诊提取
        #------------------------------------------------------#
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        #------------------------------------------------------#
        #   1x1Conv和3x3DWConv的输出拼接返回
        #------------------------------------------------------#
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)

        # 如果通道数多的话只要需要的层数
        return out[:,:self.oup,:,:]

#------------------------------------------------------#
#   倒残差结构
#   GhostModule -> BN ReLU -> GhostModule -> BN
#   GhostModule -> BN ReLU -> DWConv s=2  -> BN -> GhostModule -> BN
#   都有残差部分
#------------------------------------------------------#
class GhostBottleneck(nn.Module):
    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        #------------------------------------------------------#
        #   GhostModule提高通道数
        #------------------------------------------------------#
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        #------------------------------------------------------#
        #   步长为2使用3x3DWConv进行宽高减半
        #------------------------------------------------------#
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        #------------------------------------------------------#
        #   注意力机制
        #------------------------------------------------------#
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        #------------------------------------------------------#
        #  GhostModule降低通道数
        #------------------------------------------------------#
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        #------------------------------------------------------#
        #   是否调整短接边
        #   3x3DWConv -> BN -> 1x1Conv ->BN
        #------------------------------------------------------#
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x

class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.dropout = dropout

        #------------------------------------------------------#
        #   building first layer
        #   416,416,3 -> 208,208,16
        #------------------------------------------------------#
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        #------------------------------------------------------#
        #   building inverted residual blocks
        #   208,208,16 -> 14, 14,160
        #------------------------------------------------------#
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        #------------------------------------------------------#
        #   14, 14,160 -> 14, 14,960
        #------------------------------------------------------#
        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
        input_channel = output_channel

        self.blocks = nn.Sequential(*stages)

        #-----------------------------------------------------------#
        #   分类层
        #   14,14,960 -> 1,1,960 -> 1,1,1280 -> 1280 -> num_classes
        #-----------------------------------------------------------#
        output_channel = 1280
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
        self.act2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.blocks(x)

        x = self.global_pool(x)

        x = self.conv_head(x)
        x = self.act2(x)
        x = x.view(x.size(0), -1)
        if self.dropout > 0.:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x


def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [
        # k: 卷积核大小,表示跨特征点能力
        # t: 第一个ghostnet模块的通道数大小,倒残差结构所以它比较大
        # c: GhostBottleneck输出通道数
        # SE:是否使用注意力机制,0不使用
        # s: 步长
        # k, t, c, SE, s
        # stage1    208,208,16 -> 208,208,16
        [[3,  16,  16, 0, 1]],
        # stage2    208,208,16 -> 104,104,24
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],
        # stage3    104,104,24 -> 52, 52, 40
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],
        # stage4    52, 52, 40 -> 26, 26, 80 -> 26, 26,112
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],
        # stage5    26, 26,112 -> 14, 14,160
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]
    ]
    return GhostNet(cfgs, **kwargs)


if __name__ == "__main__":
    from torchsummary import summary

    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ghostnet().to(device)
    summary(model, input_size=(3,224,224))

