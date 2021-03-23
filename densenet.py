from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNet(nn.Module):
    """DenseNet-BC model based on the paper 'Densely Connected Convolutional Networks' (See https://arxiv.org/pdf/1608.06993.pdf).
    
    Adapted from the PyTorch implementation (See https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py).
    """
    def __init__(self, *, num_input_channels=3, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        """Instantiate the DenseNet.
        
        Parameters:
        - growth_rate should be an integer for how many filters to add each layer.
        - block_config should be a list of 4 integers for how many layers in each pooling block.
        - num_init_features should be an integer for the number of filters to learn in the first convolution layer.
        - bn_size should be an integer for the multiplicative factor for the number of bottleneck layers.
        - drop_rate should be a float for the dropout rate after each dense layer.
        - num_classes should be an integer for the number of classification classes.
        """
        super(DenseNet, self).__init__()
        
        # First convolution layer
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(num_input_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # First dense block
        block = _DenseBlock(block_config[0], num_init_features, bn_size, growth_rate, drop_rate)
        self.features.add_module('denseblock1', block)
        
        # Rest of the dense blocks
        num_features = num_init_features + block_config[0] * growth_rate
        for i, num_layers in enumerate(block_config[1:]):
            trans = _Transition(num_features, num_features//2)
            self.features.add_module('transition'+str(i+1), trans)
            num_features //= 2
            
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module('denseblock'+str(i+2), block)
            num_features += num_layers * growth_rate
        
        # Final batch norm layer
        self.features.add_module('pool'+str(i+3), nn.BatchNorm2d(num_features))
        
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Store model arguments
        self.args = {
            'num_input_channels': num_input_channels,
            'growth_rate'       : growth_rate,
            'block_config'      : block_config,
            'num_init_features' : num_init_features,
            'bn_size'           : bn_size,
            'drop_rate'         : drop_rate,
            'num_classes'       : num_classes,
        }
    
    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class _DenseBlock(nn.ModuleDict):
    """Dense block implementation for the DenseNet model."""
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features=num_input_features+i*growth_rate,
                               growth_rate=growth_rate,
                               bn_size=bn_size,
                               drop_rate=drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        x = [x]
        for _, layer in self.items():
            x.append(layer(x))
        return torch.cat(x, 1)

class _DenseLayer(nn.Module):
    """Dense layer implementation for the DenseNet model."""
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = float(drop_rate)
    
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = [x]
        x = torch.cat(x, 1)
        x = self.conv1(self.relu1(self.norm1(x)))
        x = self.conv2(self.relu2(self.norm2(x)))
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x

class _Transition(nn.Sequential):
    """Transition layer implementation for the DenseNet model."""
    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)