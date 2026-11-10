import torch
from torch import nn


# -------------------------- 1. CA 坐标注意力 --------------------------
class CA(nn.Module):
    def __init__(self, c1, c2=None, reduction=16):
        """
        Coordinate Attention 坐标注意力
        :param c1: 输入通道数
        :param c2: 输出通道数（默认等于输入通道数）
        :param reduction: 降维系数.
        """
        super().__init__()
        c2 = c2 if c2 else c1
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 高度维度池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 宽度维度池化

        mid_channels = max(8, c1 // reduction)  # 防止通道数过小
        self.conv1 = nn.Conv2d(c1, mid_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.SiLU()  # YOLOv11 默认激活

        self.conv_h = nn.Conv2d(mid_channels, c2, kernel_size=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(mid_channels, c2, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        _b, _c, h, w = x.shape
        # 沿高度/宽度池化
        x_h = self.pool_h(x)  # [b, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [b, c, w, 1]

        # 特征融合 + 降维
        y = torch.cat([x_h, x_w], dim=2)  # [b, c, h+w, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 拆分并生成注意力权重
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()  # [b, c, h, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [b, c, 1, w]

        # 加权输出
        return x * a_h * a_w


# -------------------------- 2. ECA 高效通道注意力 --------------------------
class ECA(nn.Module):
    def __init__(self, c1, c2=None, k_size=3):
        """
        Efficient Channel Attention 高效通道注意力
        :param c1: 输入通道数
        :param c2: 输出通道数（默认等于输入通道数）
        :param k_size: 一维卷积核大小.
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


# -------------------------- 3. SAM 空间注意力 --------------------------
class SAM(nn.Module):
    def __init__(self, c1, c2=None, kernel_size=7):
        """
        Spatial Attention Module 空间注意力
        :param c1: 输入通道数
        :param c2: 输出通道数（默认等于输入通道数）
        :param kernel_size: 卷积核大小.
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


# -------------------------- 4. RFA 递归特征聚合注意力 --------------------------
class RFA(nn.Module):
    def __init__(self, c1, c2=None, scale=1, kernel_size=5):
        """
        Recursive Feature Aggregation 递归特征聚合注意力
        :param c1: 输入通道数
        :param c2: 输出通道数（默认等于输入通道数）
        :param scale: 特征缩放系数
        :param kernel_size: 卷积核大小.
        """
        super().__init__()
        c2 = c2 if c2 else c1
        self.scale = scale
        self.kernel_size = kernel_size

        # 特征降维/升维
        self.reduce = nn.Conv2d(c1, c1 // scale, 1, bias=False)
        self.align = nn.Conv2d(c1 // scale, c1, 1, bias=False)

        # 递归卷积（多分支特征聚合）
        self.conv0 = nn.Conv2d(
            c1 // scale, c1 // scale, kernel_size, padding=kernel_size // 2, groups=c1 // scale, bias=False
        )
        self.conv1 = nn.Conv2d(
            c1 // scale, c1 // scale, kernel_size, padding=kernel_size // 2, groups=c1 // scale, bias=False
        )
        self.conv2 = nn.Conv2d(
            c1 // scale, c1 // scale, kernel_size, padding=kernel_size // 2, groups=c1 // scale, bias=False
        )
        self.conv3 = nn.Conv2d(
            c1 // scale, c1 // scale, kernel_size, padding=kernel_size // 2, groups=c1 // scale, bias=False
        )

        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

    def forward(self, x):
        # 降维
        x_red = self.reduce(x)
        # 递归特征聚合
        x0 = self.conv0(x_red)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # 特征融合 + 升维
        y = self.align(x0 + x1 + x2 + x3)
        # 残差连接 + 激活
        y = self.bn(y)
        y = self.act(y)
        return x + y
