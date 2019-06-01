import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import collections

__all__ = ['resnet50']



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, layer_index, block_index, cfg=None, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if cfg is None:
            cfg = [planes]
        assert len(cfg) == 1
        self.conv1 = conv3x3(inplanes, cfg[0], stride)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(cfg[0], planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.layer_index = layer_index
        self.block_index = block_index

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, layer_index, block_index,
                                        cfg=None, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        if cfg is None:
            cfg = [planes, planes]
        assert len(cfg)==2
        self.conv1 = nn.Conv2d(inplanes, cfg[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.conv2 = nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv3 = nn.Conv2d(cfg[1], planes*self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, cfg=None, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()


        cfg_layers = [None] * 4
        if cfg is not None:
            cfg_idx = [sum(layers[:i]) * 2 for i in range(len(layers)+1)]
            for i in range(4):
                cfg_layers[i] = cfg[cfg_idx[i] : cfg_idx[i+1]]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cfg_layers[0], layer_index='1')
        self.layer2 = self._make_layer(block, 128, layers[1], cfg_layers[1], layer_index='2', stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg_layers[2], layer_index='3', stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg_layers[3], layer_index='4', stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, cfg, layer_index, stride=1):
        downsample = None
        cfg_blocks = [None] * blocks
        if cfg is not None:
            len_ = len(cfg) / blocks
            assert len(cfg)%blocks==0
            for i in range(blocks):
                cfg_blocks[i] = cfg[i * len_:(i + 1) * len_]
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            layer_index = layer_index, block_index=0,
                            cfg=cfg_blocks[0],
                            stride=stride, downsample = downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                            layer_index = layer_index, block_index=i,
                            cfg=cfg_blocks[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model



