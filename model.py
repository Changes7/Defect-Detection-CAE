import torch
import torch.nn as nn

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()
        
        # --- 编码器 (Encoder): 把图片压缩成特征 ---
        self.encoder = nn.Sequential(
            # 输入: (3, 256, 256) -> 输出: (32, 128, 128)
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 输出: (64, 64, 64)
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 输出: (128, 32, 32)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True)
        )

        # --- 解码器 (Decoder): 尝试还原图片 ---
        self.decoder = nn.Sequential(
            # 输入: (128, 32, 32) -> 输出: (64, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 输出: (32, 128, 128)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            # 输出: (3, 256, 256) 还原回原始尺寸和通道
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # 使用 Sigmoid 将像素值限制在 [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# 测试模型结构
if __name__ == "__main__":
    model = ConvAutoEncoder()
    # 模拟一张图片输入: (BatchSize=1, Channel=3, Width=256, Height=256)
    test_input = torch.randn(1, 3, 256, 256)
    test_output = model(test_input)
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {test_output.shape}")
    print("模型结构验证成功！")