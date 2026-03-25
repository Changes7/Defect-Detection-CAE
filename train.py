import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import ConvAutoEncoder
import os

# --- 1. 参数设置 ---
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 数据加载 ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 假设你的数据在 data/bottle 目录下
train_dataset = datasets.ImageFolder(root='data/bottle', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. 模型初始化 ---
model = ConvAutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. 开始训练 ---
loss_history = []
print(f"正在 {device} 上开始训练...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0  # 【已修复】每一轮开始前清零
    
    for img, _ in train_loader:
        img = img.to(device)
        
        # 前向与反向传播
        output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * img.size(0)

    # 计算平均 Loss
    avg_loss = train_loss / len(train_loader.dataset)
    loss_history.append(avg_loss)
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}')

# --- 5. 保存结果 ---
torch.save(model.state_dict(), 'weights/bottle_ae.pth')

# 将 Loss 记录存入文本文件，方便画图
with open('results/loss_bn.txt', 'w') as f:
    for l in loss_history:
        f.write(f"{l}\n")

print("训练完成！模型已保存，Loss 数据已存入 loss_bn.txt")