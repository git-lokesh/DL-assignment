import torch
import torch.nn as nn

# AlexNet model
class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AlexNet, self).__init__()

        # Convolutional layers
        self.alexnet_conv1 = nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2)
        self.alexnet_conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.alexnet_conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.alexnet_conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.alexnet_conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.alexnet_fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.alexnet_fc2 = nn.Linear(4096, 4096)
        self.alexnet_fc3 = nn.Linear(4096, num_classes)

        # Other layers
        self.alexnet_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.alexnet_norm = nn.LocalResponseNorm(size=5, k=2)
        self.alexnet_dropout = nn.Dropout(0.5)
        self.alexnet_relu = nn.ReLU()

        self._initialize_weights()

    def forward(self, x):
        x = self.alexnet_pool(self.alexnet_norm(self.alexnet_relu(self.alexnet_conv1(x))))
        x = self.alexnet_pool(self.alexnet_norm(self.alexnet_relu(self.alexnet_conv2(x))))
        x = self.alexnet_relu(self.alexnet_conv3(x))
        x = self.alexnet_relu(self.alexnet_conv4(x))
        x = self.alexnet_pool(self.alexnet_relu(self.alexnet_conv5(x)))

        x = torch.flatten(x, 1)

        x = self.alexnet_dropout(self.alexnet_relu(self.alexnet_fc1(x)))
        x = self.alexnet_dropout(self.alexnet_relu(self.alexnet_fc2(x)))
        x = self.alexnet_fc3(x)
        return x

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 1 if isinstance(layer, nn.Conv2d) else 0)

# Example input 
sample_input = torch.randn(64, 3, 227, 227)
model = AlexNet(sample_input.shape[1], 1000)
output = model(sample_input)
print(output.shape) # OUTPUT : ([64, 1000])
