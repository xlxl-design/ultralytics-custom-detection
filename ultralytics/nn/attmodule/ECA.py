import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- 2. ECA 高效通道注意力 --------------------------
class ECA(nn.Module):
    def __init__(self, c1, c2=None, k_size=3):
        """
        Efficient Channel Attention 高效通道注意力
        :param c1: 输入通道数
        :param c2: 输出通道数（默认等于输入通道数）
        :param k_size: 一维卷积核大小
        """
        super().__init__()
        c2 = c2 if c2 else c1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        # 一维卷积（替代全连接层，减少参数）
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化 [b, c, h, w] -> [b, c, 1, 1]
        y = self.avg_pool(x)
        # 一维卷积处理 [b, c, 1, 1] -> [b, 1, c] -> [b, 1, c]
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # 生成通道注意力权重并加权
        y = self.sigmoid(y)
        return x * y.expand_as(x)