import torch
import torch.nn as nn
import torch.nn.functional as F


class CA(nn.Module):
    def __init__(self, c1, c2=None, reduction=16):
        super().__init__()
        c2 = c2 if c2 else c1
        self.c1 = c1
        self.c2 = c2

        # 全局池化后展平为 [b, c, h] 或 [b, c, w]
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # [b, c, h, 1] → 展平为 [b, c, h]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # [b, c, 1, w] → 展平为 [b, c, w]

        mid_channels = max(8, c1 // reduction)
        self.conv1 = nn.Conv1d(c1, mid_channels, kernel_size=1, bias=False)  # 改用1D卷积适配展平维度
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.act = nn.SiLU()

        # 输出卷积层（1D转2D，恢复空间维度）
        self.conv_h = nn.Conv1d(mid_channels, c2, kernel_size=1, bias=False)
        self.conv_w = nn.Conv1d(mid_channels, c2, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.c1, f"输入通道{c}≠初始化通道{self.c1}"

        # 1. 池化并展平为 3D 张量 [b, c, h] / [b, c, w]
        x_h = self.pool_h(x).squeeze(-1)  # [b, c, h, 1] → [b, c, h]
        x_w = self.pool_w(x).squeeze(2)  # [b, c, 1, w] → [b, c, w]

        # 2. 拼接为 [b, c, h+w]（dim=2 为长度维度，无维度冲突）
        y = torch.cat([x_h, x_w], dim=2)  # [b, c, h+w]

        # 3. 1D卷积降维（适配展平后的维度）
        y = self.conv1(y)  # [b, mid, h+w]
        y = self.bn1(y)
        y = self.act(y)

        # 4. 拆分回 h/w 两部分
        x_h, x_w = torch.split(y, [h, w], dim=2)  # [b, mid, h] / [b, mid, w]

        # 5. 生成注意力权重并恢复 4D 形状
        a_h = self.conv_h(x_h).sigmoid().unsqueeze(-1)  # [b, c2, h] → [b, c2, h, 1]
        a_w = self.conv_w(x_w).sigmoid().unsqueeze(2)  # [b, c2, w] → [b, c2, 1, w]

        # 6. 加权输出（维度完全匹配）
        return x * a_h * a_w