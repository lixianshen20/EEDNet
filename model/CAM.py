import torch
import torch.nn as nn


class eca_layer(nn.Module):
    """Efficient Channel Attention (ECA)"""

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size,
            padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)  # (b, c, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (b, 1, c)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CMA(nn.Module):
    """Connection + Multi-scale Attention"""

    def __init__(self, in_channels=128, mid_channels=64, out_channels=9, k_size=3):
        super(CMA, self).__init__()
        self.connect_branch = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=3, dilation=3),
        )
        # 加入ECA模块
        self.eca = eca_layer(out_channels, k_size=k_size)

    def forward(self, x):
        x = self.connect_branch(x)
        x = self.eca(x)
        return x
