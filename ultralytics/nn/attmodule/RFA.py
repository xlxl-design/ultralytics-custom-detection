from torch import nn


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
