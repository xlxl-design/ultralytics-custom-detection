import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 3. SAM 空间注意力 --------------------------
class SAM(nn.Module):
    def __init__(self, c1, c2=None, kernel_size=7):
        """
        Spatial Attention Module 空间注意力
        :param c1: 输入通道数
        :param c2: 输出通道数（默认等于输入通道数）
        :param kernel_size: 卷积核大小
        """
        super().__init__()
        c2 = c2 if c2 else c1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算均值和最大值 [b, c, h, w] -> [b, 1, h, w]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接特征 [b, 2, h, w]
        y = torch.cat([avg_out, max_out], dim=1)
        # 空间注意力权重生成
        y = self.conv(y)
        y = self.bn(y)
        y = self.sigmoid(y)
        # 加权输出
        return x * y.expand_as(x)