import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Inception_AttentionNet', 'SE_UWNet']

path = './se_uwnet.pkl'


def SE_UWNet(pretrained=False, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        model = Inception_AttentionNet(**kwargs)
        model.load_state_dict(torch.load(path))
        return model

    return Inception_AttentionNet(**kwargs)


class Inception_AttentionNet(nn.Module):

    def __init__(self, num_classes=2):
        super(Inception_AttentionNet, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=7, stride=2),  # 147*147*16
            BasicConv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 147*147*32
            nn.MaxPool2d(kernel_size=3, stride=2),  # 73*73*32
            Inception_1(32, 32),  # 36*36*128
            SEBasicBlock(128,128),
            Inception_2(128, 128),  # 17*17*512
            SEBasicBlock(512,512),
            Inception_3(512, 256),  # 8*8*1024
        )

        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception_1(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(Inception_1, self).__init__()
        self.branch1x1_1 = BasicConv2d(in_channels, 8, kernel_size=1)
        self.branch1x1_2 = BasicConv2d(8, 32, kernel_size=3, stride=2)

        self.branch5x5_1 = BasicConv2d(in_channels, 8, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(8, 32, kernel_size=5, padding=2)
        self.branch5x5_3 = BasicConv2d(32, 32, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 8, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(8, 32, kernel_size=7, padding=3)
        self.branch3x3dbl_3 = BasicConv2d(32, 32, kernel_size=3, stride=2)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1_1(x)
        branch1x1 = self.branch1x1_2(branch1x1)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Inception_2(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(Inception_2, self).__init__()
        self.branch1x1_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch1x1_2 = BasicConv2d(64, 128, kernel_size=3, stride=2)

        self.branch5x5_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(64, 128, kernel_size=5, padding=2)
        self.branch5x5_3 = BasicConv2d(128, 128, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 128, kernel_size=7, padding=3)
        self.branch3x3dbl_3 = BasicConv2d(128, 128, kernel_size=3, stride=2)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1_1(x)
        branch1x1 = self.branch1x1_2(branch1x1)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class Inception_3(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(Inception_3, self).__init__()
        self.branch1x1_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.branch1x1_2 = BasicConv2d(128, 256, kernel_size=3, stride=2)

        self.branch5x5_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(128, 256, kernel_size=5, padding=2)
        self.branch5x5_3 = BasicConv2d(256, 256, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(128, 256, kernel_size=7, padding=3)
        self.branch3x3dbl_3 = BasicConv2d(256, 256, kernel_size=3, stride=2)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1_1(x)
        branch1x1 = self.branch1x1_2(branch1x1)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch5x5 = self.branch5x5_3(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=2)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)




def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out






