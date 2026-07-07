"""DnCNN-lite: a small residual CNN denoiser (the learned ISP stage).

This is the "AI" half of the hybrid ISP. It plays the same role as the classical
bilateral denoiser but is *learned*: given the classical pipeline's sRGB output
(demosaiced + WB + CCM + gamma, noise still present), it predicts the residual
noise and subtracts it.

Architecture (DnCNN, Zhang et al. 2017, shrunk for CPU):
    Conv(3->C, 3x3) + ReLU
    [ Conv(C->C, 3x3) + BN + ReLU ] x (depth-2)
    Conv(C->3, 3x3)                       -> predicts residual (noise)
    output = input - residual             (residual learning)

Residual learning is the key trick: the network learns the noise, which is near
zero-mean and easier to fit than the clean image, so it converges fast on little
data. Padding keeps spatial size; the net is fully-convolutional so it trains on
patches and infers on full frames.
"""
import torch
import torch.nn as nn


class DnCNNLite(nn.Module):
    def __init__(self, channels=3, features=32, depth=6):
        super().__init__()
        layers = [nn.Conv2d(channels, features, 3, padding=1, bias=True),
                  nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(features, features, 3, padding=1, bias=False),
                       nn.BatchNorm2d(features),
                       nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, channels, 3, padding=1, bias=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.net(x)      # predict the noise
        return x - residual         # clean = noisy - noise
