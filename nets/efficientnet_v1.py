import copy
import math
import torch
from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, List, Optional, Sequence
from torch.hub import load_state_dict_from_url

from torchvision.ops import StochasticDepth # DropPath层


__all__ = ["EfficientNet", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"]


model_urls = {
    # Weights ported from https://github.com/rwightman/pytorch-image-models/
    "efficientnet_b0": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth",
    "efficientnet_b1": "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth",
    "efficientnet_b2": "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth",
    "efficientnet_b3": "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth",
    "efficientnet_b4": "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth",
    # Weights ported from https://github.com/lukemelas/EfficientNet-PyTorch/
    "efficientnet_b5": "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth",
    "efficientnet_b6": "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth",
    "efficientnet_b7": "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth",
}


def _make_divisible(v: float, divisor: int, min_value: int = None) -> int:
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
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvNormActivation(nn.Sequential):
    '''
    标准卷积块
    Conv + BN + ReLU
    默认宽高不变
    '''
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        stride:       int = 1,
        padding:      int = None,
        groups:       int = 1,
        norm_layer       = nn.BatchNorm2d, # BN,默认BatchNorm2d
        activation_layer = nn.ReLU,        # 激活函数,默认ReLU
        dilation:     int = 1,
        inplace:      bool = True,
    ) -> None:
        # 自动调整padding,让宽高不变
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels

#---------------------------------------------------#
#   注意力机制
#   两个1x1Conv代替全连接层,不需要变换维度
#       对特征矩阵每一个channel进行池化,得到长度为channel的一维向量,使用两个全连接层,
#       两个线性层的长度,最后得到权重,然后乘以每层矩阵的原值
#           线性层长度变化: channel -> channel / 4 -> channel
#---------------------------------------------------#
class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        input_channels:   int,  # in_channels&out_channels
        squeeze_channels: int,  # 中间维度,是输入block维度的1/4
        activation       = nn.ReLU,
        scale_activation = nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 两个卷积作为全连接层,kernel为1
        self.fc1              = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.activation       = activation()        # SiLU 别名 swish
        self.fc2              = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.scale_activation = scale_activation()  # sigmoid

    def _scale(self, input: Tensor) -> Tensor:
        # [batch, channel, height, width] -> [batch, channel, 1, 1]
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        # [batch, channel, 1, 1] * [batch, channel, height, width]
        # 高维度矩阵相乘是最后两个维度相乘,所以是 [1, 1] 点乘 [h, w]
        return scale * input


class MBConvConfig:
    '''
    每一个stage中所有的MBConv配置参数
    '''
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(self,
                expand_ratio:   float, # MBConv第一层 1x1 Conv扩展比率 1 or 6
                kernel:         int,
                stride:         int,
                input_channels: int,   # in_channel
                out_channels:   int,   # out_channel
                num_layers:     int,   # MBConv重复次数
                width_mult:     float, # 开始和最终维度调整倍率
                depth_mult:     float  # MBConv重复次数调整倍率
                ) -> None:

        self.expand_ratio = expand_ratio    # MBConv第一层 1x1 Conv扩展比率 1 or 6
        self.kernel = kernel
        self.stride = stride                # DW卷积步长 1 or 2
        self.input_channels = self.adjust_channels(input_channels, width_mult)  # 开始维度调整倍率
        self.out_channels   = self.adjust_channels(out_channels,   width_mult)  # 最终维度调整倍率
        self.num_layers     = self.adjust_depth(   num_layers,     depth_mult)  # MBConv重复次数调整倍率

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'expand_ratio={expand_ratio}'
        s += ', kernel={kernel}'
        s += ', stride={stride}'
        s += ', input_channels={input_channels}'
        s += ', out_channels={out_channels}'
        s += ', num_layers={num_layers}'
        s += ')'
        return s.format(**self.__dict__)

    @staticmethod
    def adjust_channels(channels: int, width_mult: float, min_value: Optional[int] = None) -> int:
        return _make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return int(math.ceil(num_layers * depth_mult))


class MBConv(nn.Module):
    '''
    倒残差
    1x1Conv => 3x3/5x5DWConv => 1x1Conv
    最后的1x1Conv没有激活函数
    只有当stride == 1 且 in_channel == out_channel 才使用shortcut连接
    '''
    def __init__(self,
                    cnf:                   MBConvConfig,
                    stochastic_depth_prob: float,
                    norm_layer,
                    se_layer = SqueezeExcitation
                ) -> None:
        super().__init__()

        # 判断每一层步长是否为1或2
        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        # shortcut连接  只有当stride == 1 且n_channel == out_channel才使用
        self.use_res_connect = (cnf.stride == 1) and (cnf.input_channels == cnf.out_channels)

        layers: List[nn.Module] = []

        # 激活函数
        activation_layer = nn.SiLU

        # expand 维度扩展,倍率为: 1 or 6
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio)

        # 1x1
        # 扩展倍率因子是否为1(扩展维度是否等于in_channel), 第一层为1就不需要使用第一个 1x1 的卷积层 Stage2的倍率不变,所以不需要它
        if expanded_channels != cnf.input_channels:
            layers.append(ConvNormActivation(cnf.input_channels,
                                            expanded_channels,
                                            kernel_size     =1,
                                            norm_layer      =norm_layer,
                                            activation_layer=activation_layer))

        # 3x3 or 5x5 depthwise  groups == in_channels == out_channels
        layers.append(ConvNormActivation(expanded_channels,
                                        expanded_channels,
                                        kernel_size     =cnf.kernel,
                                        stride          =cnf.stride,
                                        groups          =expanded_channels,   # groups == in_channels == out_channels
                                        norm_layer      =norm_layer,
                                        activation_layer=activation_layer))

        # 注意力机制
        # 中间维度为初始维度的 1/4,而不是扩展维度的 1/4
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(se_layer(expanded_channels, squeeze_channels, activation=partial(nn.SiLU, inplace=True)))

        # 1x1Conv 不要激活函数
        layers.append(ConvNormActivation(expanded_channels,
                                        cnf.out_channels,
                                        kernel_size     =1,
                                        norm_layer      =norm_layer,
                                        activation_layer=None))

        self.block = nn.Sequential(*layers)

        # DropPath层
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels     = cnf.out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)

        # 只有在使用shortcut连接时才使用DropPath层
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(nn.Module):
    def __init__(
            self,
            inverted_residual_setting: List[MBConvConfig],  # 倒残差配置
            dropout:                float,                  # 最终线性层的Dropout参数
            stochastic_depth_prob:  float = 0.2,            # Droppath的drop参数
            num_classes:            int   = 1000,           # 分类数
            block                         = None,           # 使用的MBConv
            norm_layer                    = None,           # BN层
            **kwargs: Any
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                    all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")

        # block默认是MBConv
        if block is None:
            block = MBConv

        # norm_layer默认是BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        #-----------------------------------#
        #   第一层卷积
        #   stem 3x3Conv+BN+Act 调整通道,宽高减半
        #-----------------------------------#
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvNormActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer, activation_layer=nn.SiLU))

        # building inverted residual blocks
        # 统计重复次数
        total_stage_blocks = sum([cnf.num_layers for cnf in inverted_residual_setting])
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                # 数据复制,为了不影响源数据
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                # 第一个MBConv的通道和步长会变化,后面的不会
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # dropout比例逐渐增大
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # 计算最后的输入 stage8的输出
        lastconv_input_channels = inverted_residual_setting[-1].out_channels    # 320
        lastconv_output_channels = 4 * lastconv_input_channels                  # 320 * 4 = 1280
        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.SiLU))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _efficientnet_conf(width_mult: float,   # 开始和最终维度调整倍率
                        depth_mult: float,  # MBConv重复次数调整倍率
                        **kwargs: Any) -> List[MBConvConfig]:
    '''
    生成MBConvConfig的参数

    MBConvConfig的参数:
        expand_ratio: float,   # MBConv第一层 1x1 Conv扩展比率 1 or 6
        kernel: int,
        stride: int,
        input_channels: int,   # in_channel
        out_channels: int,     # out_channel
        num_layers: int,       # MBConv重复次数
        width_mult: float,     # 开始和最终维度调整倍率
        depth_mult: float      # MBConv重复次数调整倍率
    '''
    # MBConvConfig后2个参数
    bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)

    #                                       Conv1 224,224, 3 -> 112,112,32   512,512, 3 -> 256,256,32
    # MBConvConfig前6个参数
    inverted_residual_setting = [
        # 扩展倍率, kernel, stride, input, out, MBConv重复次数
        bneck_conf(1, 3, 1, 32,  16,  1),       # 112,112,32 -> 112,112,16   256,256,32 -> 256,256,16
        bneck_conf(6, 3, 2, 16,  24,  2),       # 112,112,16 -> 56, 56, 24   256,256,16 -> 128,128,24   s=2
        bneck_conf(6, 5, 2, 24,  40,  2),       # 56, 56, 24 -> 28, 28, 40   128,128,24 -> 64, 64, 40   s=2 out_channels: 40 40 48 48 56 64 72 80
        bneck_conf(6, 3, 2, 40,  80,  3),       # 28, 28, 40 -> 14, 14, 80   64, 64, 40 -> 32, 32, 80   s=2
        bneck_conf(6, 5, 1, 80,  112, 3),       # 14, 14, 80 -> 14, 14,112   32, 32, 80 -> 32, 32,112   out_channels: 112 112 120 136 160 176 200 224
        bneck_conf(6, 5, 2, 112, 192, 4),       # 14, 14,112 -> 7,  7, 192   32, 32,112 -> 16, 16,192   s=2
        bneck_conf(6, 3, 1, 192, 320, 1),       # 7 , 7, 192 -> 7,  7, 320   16, 16,192 -> 16, 16,320   out_channels: 320 320 352 384 448 512 576 640
    ]

    return inverted_residual_setting


def _efficientnet_model(
    arch: str,
    inverted_residual_setting: List[MBConvConfig],  # 多个
    dropout: float,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> EfficientNet:
    model = EfficientNet(inverted_residual_setting, dropout, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def efficientnet_b0(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    224x224

    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.0, depth_mult=1.0, **kwargs)
    return _efficientnet_model("efficientnet_b0", inverted_residual_setting, 0.2, pretrained, progress, **kwargs)


def efficientnet_b1(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    240x240

    Constructs a EfficientNet B1 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.0, depth_mult=1.1, **kwargs)
    return _efficientnet_model("efficientnet_b1", inverted_residual_setting, 0.2, pretrained, progress, **kwargs)


def efficientnet_b2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    260x260

    Constructs a EfficientNet B2 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.1, depth_mult=1.2, **kwargs)
    return _efficientnet_model("efficientnet_b2", inverted_residual_setting, 0.3, pretrained, progress, **kwargs)


def efficientnet_b3(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    300x300

    Constructs a EfficientNet B3 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.2, depth_mult=1.4, **kwargs)
    return _efficientnet_model("efficientnet_b3", inverted_residual_setting, 0.3, pretrained, progress, **kwargs)


def efficientnet_b4(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    380x380

    Constructs a EfficientNet B4 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.4, depth_mult=1.8, **kwargs)
    return _efficientnet_model("efficientnet_b4", inverted_residual_setting, 0.4, pretrained, progress, **kwargs)


def efficientnet_b5(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    456x456

    Constructs a EfficientNet B5 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.6, depth_mult=2.2, **kwargs)
    return _efficientnet_model("efficientnet_b5", inverted_residual_setting, 0.4, pretrained, progress,
                               norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), **kwargs)


def efficientnet_b6(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    528x528

    Constructs a EfficientNet B6 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.8, depth_mult=2.6, **kwargs)
    return _efficientnet_model("efficientnet_b6", inverted_residual_setting, 0.5, pretrained, progress,
                               norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), **kwargs)


def efficientnet_b7(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> EfficientNet:
    """
    600x600

    Constructs a EfficientNet B7 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=2.0, depth_mult=3.1, **kwargs)
    return _efficientnet_model("efficientnet_b7", inverted_residual_setting, 0.5, pretrained, progress,
                               norm_layer=partial(nn.BatchNorm2d, eps=0.001, momentum=0.01), **kwargs)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = efficientnet_b0(pretrained=False)
    features = model.features

    x = torch.rand(1, 3, 416, 416)

    x = features[0:4](x)
    print(x.size())         # 40  40  48  48  56  64  72  80

    x = features[4:6](x)    # 112 112 120 136 160 176 200 224
    print(x.size())

    x = features[6:8](x)    # 320 320 352 384 448 512 576 640
    print(x.size())
