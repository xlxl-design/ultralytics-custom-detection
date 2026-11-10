import torch
import torch.nn as nn

class CBAM(nn.Module):
    """CBAM注意力模块（通道+空间双注意力，适配YOLO11，接口和CA一致）"""
    def __init__(self, c1, c2=None, reduction=16, kernel_size=7):
        # c1: 输入通道数, c2: 输出通道数（默认和输入一致）, reduction: 通道降维系数, kernel_size: 空间注意力卷积核
        super().__init__()
        if c2 is None:
            c2 = c1
        self.c = c1
        
        # 1. 通道注意力（和SE类似，增加最大池化分支）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, c1 // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c1 // reduction, c2, bias=False)
        )
        
        # 2. 空间注意力（聚焦小目标位置特征）
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 通道注意力：平均池化+最大池化分支融合
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        x = x * channel_att  # 通道加权
        
        # 空间注意力：对特征图做平均/最大池化，融合后卷积加权
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att  # 空间加权
        
        return x  # 输出通道数和输入一致，适配YOLO插层