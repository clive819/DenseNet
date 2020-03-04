from config import DenseNetConfig as dnc
import torch
import torch.nn as nn


class _transitionLayer(nn.Module):
    def __init__(self, inChannels):
        super(_transitionLayer, self).__init__()

        self.outChannels = int(inChannels * dnc.compressionRate)

        self.module = nn.Sequential(
            nn.GroupNorm(dnc.numGroups, inChannels),
            nn.ReLU(),
            nn.Conv2d(inChannels, self.outChannels, 1),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.module(x)


class _convBlock(nn.Module):
    def __init__(self, inChannels):
        super(_convBlock, self).__init__()

        self.outChannels = dnc.growthRate
        self.module = nn.Sequential(
            nn.GroupNorm(dnc.numGroups, inChannels),
            nn.ReLU(),
            nn.Conv2d(inChannels, 4 * dnc.growthRate, 1),
            nn.GroupNorm(dnc.numGroups, 4 * dnc.growthRate),
            nn.ReLU(),
            nn.Conv2d(4 * dnc.growthRate, dnc.growthRate, 3, padding=1)
        )

    def forward(self, x):
        return self.module(x)


class _denseBlock(nn.Module):
    def __init__(self, inChannels, numBlocks):
        super(_denseBlock, self).__init__()

        self.outChannels = inChannels

        self.layers = nn.ModuleList()
        for _ in range(numBlocks):
            self.layers.append(_convBlock(self.outChannels))
            self.outChannels += dnc.growthRate

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, 1)))

        return torch.cat(features, 1)


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        outChannels = 64

        self.input = nn.Sequential(
            nn.Conv2d(3, outChannels, 7, padding=3),
            nn.GroupNorm(dnc.numGroups, outChannels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        layers = [self.input]

        for num in dnc.numBlocks:
            block = _denseBlock(outChannels, num)
            outChannels = block.outChannels
            trans = _transitionLayer(outChannels)
            outChannels = trans.outChannels
            layers.append(block)
            layers.append(trans)

        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)
