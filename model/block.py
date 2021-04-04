import torch.nn as nn


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)


class BasicBlock1x1(nn.Module):
    expansion = 1

    def __init__(self, inplanes1, planes, conv_dropout_ratio=0.2, stride=1, downsample=None):
        super(BasicBlock1x1, self).__init__()
        self.conv1 = conv1x1(inplanes1, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.RReLU()
        self.conv2 = conv1x1(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.conv_dropout = nn.Dropout(p=conv_dropout_ratio)

    def forward(self, x):
        residual = x

        """ 
        # full pre-activation
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.conv_dropout(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.conv_dropout(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out
        """

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv_dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv_dropout(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, conv_dropout_ratio=0.2, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.RReLU()
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.conv_dropout = nn.Dropout(p=conv_dropout_ratio)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv_dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv_dropout(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class BasicBlock5x5(nn.Module):
    expansion = 1

    def __init__(self, inplanes5, planes, conv_dropout_ratio=0.2, stride=1, downsample=None):
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.RReLU()
        self.conv2 = conv5x5(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.conv_dropout = nn.Dropout(p=conv_dropout_ratio)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv_dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv_dropout(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class BasicBlock7x7(nn.Module):
    expansion = 1

    def __init__(self, inplanes7, planes, conv_dropout_ratio=0.2, stride=1, downsample=None):
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.RReLU()
        self.conv2 = conv1x1(planes, planes, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.conv_dropout = nn.Dropout(p=conv_dropout_ratio)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv_dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv_dropout(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # d = residual.shape[2] - out.shape[2]
        # out1 = residual[:, :, 0:-d] + out
        # out1 = self.relu(out1)
        # return out1

        out += residual
        out = self.relu(out)
        return out
