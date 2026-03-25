import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import ConvAutoEncoder  # 导入你刚写的模型
import os

# --- 1. 超参数设置 ---
EPOCHS = 50           # 训练轮数（先跑50轮看看效果）
BATCH_SIZE = 16       # 每批处理的照片数量
LEARNING_RATE = 1e-3  # 学习率

# --- 2. 数据预处理与加载 ---
# 既然是 AI 训练，我们需要把图片统一大小并转化为 Tensor
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 加载训练集（只用 good 文件夹里的正常瓶子）
train_dataset = datasets.ImageFolder(root='./data/metal_nut/train', transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. 初始化模型、损失函数和优化器 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoEncoder().to(device)

# 使用均方误差损失（MSE），计算原图和还原图之间的差距
criterion = nn.MSELoss() 
# 使用 Adam 优化器，自动调整学习步伐
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. 开始训练循环 ---
print(f"开始在 {device} 上训练...")

for epoch in range(EPOCHS):
    train_loss = 0.0
    for data in train_loader:
        img, _ = data  # 只要图片，不要标签（自编码器是无监督学习）
        img = img.to(device)

        # 前向传播：图片进，还原图出
        output = model(img)
        # 计算差距（原图 vs 还原图）
        loss = criterion(output, img)

        # 反向传播：让 AI 纠错
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * img.size(0)
    
    # 每轮打印一次平均损失
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}')

# --- 5. 保存训练好的“大脑” ---
torch.save(model.state_dict(), 'bottle_ae.pth')
print("训练完成！模型已保存为 bottle_ae.pth")