"""Resnet Network"""
import torch
import torch.nn as nn
from model.block import BasicBlock1x1, BasicBlock3x3, BasicBlock5x5, BasicBlock7x7


class CNN(torch.nn.Module):
    def __init__(self, output_size, feature_num, input_channel = 1, conv_dropout_ratio = 0.4, fc_dropout_ratio = 0.4, 
                 channel1 = 32, channel2 = 64, channel3 = 64, dense_size = 128, layers = [1, 1, 1, 1],
                 chunk_depth = 5):
        """
        """
        
        self.channel3    = channel3
        self.chunk_depth = chunk_depth
        super(CNN, self).__init__()
        
        self.dr   = nn.Dropout(p = fc_dropout_ratio)
        self.lsm   = nn.LogSoftmax(dim=1)
        
        self.conv1 = nn.Sequential(
                    nn.Conv2d( in_channels = input_channel, out_channels = channel1, kernel_size = 3, padding = 1)
                  , nn.PReLU(init = 0)
                )
        self.conv2 = nn.Sequential(
                    nn.Conv2d( in_channels = channel1, out_channels = channel2, kernel_size = 3, padding = 1)
                  , nn.PReLU(init = 0)
                )
        self.conv3 = nn.Sequential(
                    nn.Conv2d( in_channels = channel2, out_channels = channel3, kernel_size = 3, padding = 1)
                  , nn.PReLU(init = 0)
                )
        
        self.fc1   = nn.Sequential(
              nn.Linear( in_features = channel3 * chunk_depth * feature_num, out_features = dense_size)
            , nn.PReLU(init = 0)
            , nn.Dropout(p = fc_dropout_ratio)
            )
        self.fc2   = nn.Sequential(
              nn.Linear( in_features = dense_size, out_features = output_size)
#            , nn.Softmax()
            )
        
    def forward(self, x):
        # Convolution
        x = self.conv1( x )
        x = self.conv2( x )
        x = self.conv3( x )

        # flattern
        x = x.view( x.size(0), -1 )
#        x = self.dr( x )
        x = self.fc1( x )
        x = self.fc2( x )
        x = self.lsm( x )
        
        return( x )

# =============================================================================
# 
# =============================================================================
        
class ResnetModel(torch.nn.Module):
    def __init__(self, output_size, feature_num = None, input_channel = 1, conv_dropout_ratio=0.2, fc_dropout_ratio=0.5,
                 channel1=32, channel2=64, channel3=128, layers=[1, 1, 1, 1],
                 chunk_depth=7 ):
        """
        Definition of the model.

        :param input_channel: input_height == window size, input_width == num of input features, so channel must be 1
        :param conv_dropout_ratio: dropout ratio for convolutional layers
        :param fc_dropout_ratio: dropout ratio for dense layers
        :param channel1: num of kernels in the 1st block
        :param channel2: num of kernels in the 2nd block
        :param channel3: num of kernels in the 3rd block
        :param layers: number of layers in each block
        :param chunk_depth: a.k.a window size
        """
        self.inplanes1 = 1
        self.inplanes3 = 1
        self.inplanes5 = 1
        self.inplanes7 = 1

        super(ResnetModel, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, channel1, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(channel1)
        self.relu  = nn.RReLU()
        self.lsm   = nn.LogSoftmax(dim = 1)
        self.softmax = nn.Softmax(dim = 1)

        self.layer1x1_1 = self._make_layer1(BasicBlock1x1, channel1, layers[0],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=1)
        self.layer1x1_2 = self._make_layer1(BasicBlock1x1, channel2, layers[1],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=1)
        self.layer1x1_3 = self._make_layer1(BasicBlock1x1, channel3, layers[2],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=1)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)

        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, channel1, layers[0],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, channel2, layers[1],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, channel3, layers[2],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=2)
        self.avgpool3 = nn.AdaptiveAvgPool2d(1)

        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, channel1, layers[0],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=3)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, channel2, layers[1],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=3)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, channel3, layers[2],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=3)
        self.avgpool5 = nn.AdaptiveAvgPool2d(1)

        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, channel1, layers[0],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=4)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, channel2, layers[1],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=4)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, channel3, layers[2],
                                            conv_dropout_ratio=conv_dropout_ratio, stride=4)
        self.avgpool7 = nn.AdaptiveAvgPool2d(1)

        self.conv_dropout = nn.Dropout(p=conv_dropout_ratio)
        self.fc_dropout = nn.Dropout(p=fc_dropout_ratio)
        self.fc = nn.Linear(channel3 * 3, output_size)

    # make blocks with kernel size 1
    def _make_layer1(self, block, planes, blocks, conv_dropout_ratio=0.2, stride=1):
        downsample = None
        if stride != 1 or self.inplanes1 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes1, planes * block.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes1, planes, conv_dropout_ratio=conv_dropout_ratio,
                            stride=stride, downsample=downsample))
        self.inplanes1 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes1, planes))

        return nn.Sequential(*layers)

    # make blocks with kernel size 3
    def _make_layer3(self, block, planes, blocks, conv_dropout_ratio=0.2, stride=1):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes3, planes, conv_dropout_ratio=conv_dropout_ratio,
                            stride=stride, downsample=downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    # make blocks with kernel size 5
    def _make_layer5(self, block, planes, blocks, conv_dropout_ratio=0.2, stride=1):
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes5, planes * block.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes5, planes, conv_dropout_ratio=conv_dropout_ratio,
                            stride=stride, downsample=downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    # make blocks with kernel size 7
    def _make_layer7(self, block, planes, blocks, conv_dropout_ratio=0.2, stride=1):
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes7, planes * block.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes7, planes, conv_dropout_ratio=conv_dropout_ratio,
                            stride=stride, downsample=downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        """
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        # x0 = self.conv_dropout(x0)
        x0 = self.relu(x0)
        """

        # """
        x = self.layer1x1_1(x0)
        x = self.layer1x1_2(x)
        x = self.layer1x1_3(x)
        x = self.avgpool1(x)
        # """

        # """
        y = self.layer3x3_1(x0)
        y = self.layer3x3_2(y)
        y = self.layer3x3_3(y)
        y = self.avgpool3(y)
        # """

        # """
        # layeys with kernel size (5, 5) seems to be redundant
        # because of (7, 7) layers below, so this isn't used anymore
        # z = self.layer5x5_1(x0)
        # z = self.layer5x5_2(z)
        # z = self.layer5x5_3(z)
        # z = self.avgpool5(z)
        # """

        # """
        k = self.layer7x7_1(x0)
        k = self.layer7x7_2(k)
        k = self.layer7x7_3(k)
        k = self.avgpool7(k)
        # """

        out = torch.cat([x, y, k], dim=1)
        out = out.squeeze()
        out = self.fc(self.fc_dropout(out))
        
        out = self.lsm(out)
#        out = self.softmax( out )
        
        return out
