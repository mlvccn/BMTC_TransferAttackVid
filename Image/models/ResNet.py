# models.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义 3x3 卷积
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 卷积，带有可选的步幅和填充"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# 定义 1x1 卷积
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 定义 BasicBlock，用于 ResNet18 和 ResNet34
class BasicBlock(nn.Module):
    expansion = 1  # 通道数的扩张倍数

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个卷积层
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        # 下采样层（如果需要）
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = F.relu(out, inplace=True)

        return out

# 定义 Bottleneck，用于 ResNet50、ResNet101 和 ResNet152
class Bottleneck(nn.Module):
    expansion = 4  # 通道数的扩张倍数

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 第一个 1x1 卷积
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        # 第二个 3x3 卷积
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        # 第三个 1x1 卷积
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        # 下采样层（如果需要）
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x  # 残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # 残差连接
        out = F.relu(out, inplace=True)

        return out

# 定义 ResNet 主体
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        # 最大池化层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 四个阶段的卷积层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 零初始化最后一个 BN 层
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # 构建层的方法
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 如果输入通道数和输出通道数不一致，或者步幅不为 1，需要下采样
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 第一个块，可能需要下采样
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        # 剩余的块
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    # 前向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 定义 ResNet 的构建函数
def resnet18(num_classes=1000, zero_init_residual=False):
    """构建 ResNet18 模型"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, zero_init_residual)

def resnet34(num_classes=1000, zero_init_residual=False):
    """构建 ResNet34 模型"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, zero_init_residual)

def resnet50(num_classes=1000, zero_init_residual=False):
    """构建 ResNet50 模型"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, zero_init_residual)

def resnet101(num_classes=1000, zero_init_residual=False):
    """构建 ResNet101 模型"""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, zero_init_residual)

def resnet152(num_classes=1000, zero_init_residual=False):
    """构建 ResNet152 模型"""
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, zero_init_residual)
