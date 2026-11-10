import torch
import torch.nn as nn

class SE(nn.Module):
    """极简版SE注意力模块（适配YOLO11，可直接在yaml中调用）"""
    def __init__(self, c1, c2=None, reduction=16):
        # c1: 输入通道数, c2: 输出通道数（默认和输入一致）, reduction: 降维系数
        super().__init__()
        if c2 is None:
            c2 = c1
        self.c = c1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // reduction, c2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        # 挤压+激励+加权，输出通道数和输入一致
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # 直接加权，无额外维度变化