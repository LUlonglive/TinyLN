import torch
import torch.nn as nn
import torch.nn.functional as F  

class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        
        # 第一个块
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu1 = nn.GELU()
        
        # 第二个块
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.gelu2 = nn.GELU()
    
    def forward(self, x):
        # 第一次运行Conv1D + BN + GELU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.gelu1(x)
        
        # 第二次运行Conv1D + BN + GELU
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.gelu2(x)
        
        return x

class IRB(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super(IRB, self).__init__()
        
        self.use_residual = in_channels == out_channels
        expanded_channels = in_channels * expansion_factor
        
        # 1x1 扩展卷积
        self.expand = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=expanded_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(expanded_channels),
            nn.GELU()
        )
        
        # 深度卷积
        self.depthwise = nn.Sequential(
            nn.Conv1d(
                in_channels=expanded_channels,
                out_channels=expanded_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=expanded_channels  # 使其成为深度卷积
            ),
            nn.BatchNorm1d(expanded_channels),
            nn.GELU()
        )
        
        # 1x1 投影卷积
        self.project = nn.Sequential(
            nn.Conv1d(
                in_channels=expanded_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x):
        identity = x
        
        # 扩展
        x = self.expand(x)
        
        # 深度卷积
        x = self.depthwise(x)
        
        # 投影
        x = self.project(x)
        
        # 残差连接
        if self.use_residual:
            x = x + identity
            
        return x

class MSDPM(nn.Module):
    def __init__(self, channels):
        super(MSDPM, self).__init__()
        
        # 初始深度卷积 (kernel_size=7)
        self.initial_dw = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                padding=3,
                groups=channels
            ),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        
        # 扩张卷积分支1 (dilation=2)
        self.dilated_conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=2,
                dilation=2
            ),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        
        # 扩张卷积分支2 (dilation=3)
        self.dilated_conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=3,
                dilation=3
            ),
            nn.BatchNorm1d(channels),
            nn.GELU()
        )
        
        # 最终1x1卷积
        self.final_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=1
            ),
            nn.BatchNorm1d(channels)
        )
        
        self.gelu = nn.GELU()
        
    def forward(self, x):
        # 初始深度卷积
        y = self.initial_dw(x)
        
        # 并行扩张卷积
        z2 = self.dilated_conv1(y)
        z3 = self.dilated_conv2(y)
        
        # 合并分支并激活
        z = self.gelu(z2 + z3)
        
        # 最终1x1卷积
        out = self.final_conv(z)
        return out

class LK_FFN(nn.Module):
    def __init__(self, channels, expansion_factor=4):
        super(LK_FFN, self).__init__()
        
        hidden_dim = channels * expansion_factor
        
        # 深度卷积 (kernel_size=7)
        self.dw_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=7,
                padding=3,
                groups=channels
            ),
            nn.BatchNorm1d(channels)
        )
        
        # 两层全连接层
        self.fc1 = nn.Conv1d(channels, hidden_dim, 1)  # 1x1 conv作为全连接层
        self.fc2 = nn.Conv1d(hidden_dim, channels, 1)  # 1x1 conv作为全连接层
        self.bn = nn.BatchNorm1d(channels)
        self.gelu = nn.GELU()
        
    def forward(self, z):
        # 深度卷积
        x = self.dw_conv(z)
        
        # 全连接层
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.bn(x)
        
        return x

class DCB(nn.Module):
    def __init__(self, channels, expansion_factor=4):
        super(DCB, self).__init__()
        
        self.msdpm = MSDPM(channels)
        self.lk_ffn = LK_FFN(channels, expansion_factor)
        
    def forward(self, x):
        # MSDPM模块
        x = x + self.msdpm(x)  # 残差连接
        
        # Large Kernel FFN模块
        x = x + self.lk_ffn(x)  # 残差连接
        
        return x

class TinyLN(nn.Module):
    def __init__(self,
                 in_channels=1,      # 输入通道数
                 stem_channels=64,    # Stem输出通道数
                 num_classes=2,       # 分类数量
                 expansion_factor=4,   # 扩展因子
                 num_irb_blocks=1,     # Inverted Residual Block 数量
                 num_dcb_blocks=1      # DCB 模块数量
                ):
        super(TinyLN, self).__init__()
        
        # Stem层进行初始特征提取
        self.stem = Stem(in_channels, stem_channels)
        
        # Inverted Residual Blocks
        self.irb_blocks = nn.Sequential()
        for i in range(num_irb_blocks):
            self.irb_blocks.add_module(
                f"irb_block_{i + 1}",
                IRB(
                    in_channels=stem_channels,
                    out_channels=stem_channels,
                    expansion_factor=expansion_factor
                )
            )
        
        # DCB 模块
        self.dcb_blocks = nn.Sequential()
        for i in range(num_dcb_blocks):
            self.dcb_blocks.add_module(
                f"dcb_block_{i + 1}",
                DCB(
                    channels=stem_channels,
                    expansion_factor=expansion_factor
                )
            )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(stem_channels, num_classes)
        )
        
    def forward(self, x, return_features=False):
        # Stem特征提取
        x = self.stem(x)
        
        # Inverted Residual Blocks
        for irb_block in self.irb_blocks:
            x = irb_block(x)
        
        # DCB 模块
        for dcb_block in self.dcb_blocks:
            x = dcb_block(x)
        
        # 全局平均池化
        x = self.global_pool(x)
        
        # 展平特征
        features = x.view(x.size(0), -1)
        
        # 分类
        output = self.classifier(features)
        
        if return_features:
            return output, features
        return output

if __name__ == "__main__":
    # 创建模型实例，例如 1 个 DCB 模块和 1 个 IRB 模块
    model = TinyLN(
        in_channels=1,
        stem_channels=64,
        num_classes=2,
        expansion_factor=4,
        num_irb_blocks=1,
        num_dcb_blocks=1
    )

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 将模型移动到设备
    model = model.to(device)

    # 假设有一个输入样本
    x = torch.randn(32, 1, 1024).to(device)  # 将输入数据移动到设备

    output, features = model(x, return_features=True)
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")

