import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(ResNetBlock, self).__init__()
        self.expansion = 4
        self.resnet_conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.resnet_bn1 = nn.BatchNorm2d(intermediate_channels)
        self.resnet_conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.resnet_bn2 = nn.BatchNorm2d(intermediate_channels)
        self.resnet_conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.resnet_bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.resnet_relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_conv2(x)
        x = self.resnet_bn2(x)
        x = self.resnet_relu(x)
        x = self.resnet_conv3(x)
        x = self.resnet_bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.resnet_relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.resnet_conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet_bn1 = nn.BatchNorm2d(64)
        self.resnet_relu = nn.ReLU()
        self.resnet_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet architecture layers
        self.resnet_layer1 = self._make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.resnet_layer2 = self._make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.resnet_layer3 = self._make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.resnet_layer4 = self._make_layer(block, layers[3], intermediate_channels=512, stride=2)

        self.resnet_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.resnet_fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.resnet_conv1(x)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)
        x = self.resnet_layer1(x)
        x = self.resnet_layer2(x)
        x = self.resnet_layer3(x)
        x = self.resnet_layer4(x)

        x = self.resnet_avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.resnet_fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride))
        self.in_channels = intermediate_channels * 4

        for _ in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(ResNetBlock, [3, 4, 6, 3], img_channel, num_classes)


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(ResNetBlock, [3, 4, 23, 3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(ResNetBlock, [3, 8, 36, 3], img_channel, num_classes)


def test():
    BATCH_SIZE = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet101(img_channel=3, num_classes=1000).to(device)
    y = net(torch.randn(BATCH_SIZE, 3, 224, 224)).to(device)
    assert y.size() == torch.Size([BATCH_SIZE, 1000])
    print(y.size())


if __name__ == "__main__":
    test()
# OUTPUT : ([4 ,1000])