from resnetBlock import ResNetBlock
import torch.nn as nn
import torch
class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()

        # Initial Convolution Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.GELU()

        # ResNet Blocks
        self.res_block1 = ResNetBlock(64, 128, stride=2)  # Downsample
        self.res_block2 = ResNetBlock(128, 256, stride=2) # Downsample
        self.res_block3 = ResNetBlock(256, 512, stride=2) # Downsample
        self.res_block4 = ResNetBlock(512,512, stride=2)

        #remove when low perf

        # Global Average Pooling and Fully Connected Layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Outputs (batch, 512, 1, 1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        x = self.avgpool(x)  
        x = torch.flatten(x, 1)  # Flatten for FC layer
        x = self.fc(x)
        return x
 
    

    
      

    