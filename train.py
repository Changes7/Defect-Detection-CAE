import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import ConvAutoEncoder
import os
from pytorch_msssim import ssim

# 【新增】安全检查：自动创建文件夹，防止运行到最后一步因为没文件夹而报错闪退
for path in ['weights', 'results', 'data']:
    os.makedirs(path, exist_ok=True)

# --- 1. 参数设置 ---
EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
# 【新增 2】设置权重比例，MSE占大头负责整体亮度，SSIM占小头负责纹理细节
ALPHA = 0.8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 数据加载 ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 【修改 1】：数据路径指向金属螺母 (Metal Nut)
# ⚠️ 注意：ImageFolder 要求子目录结构，确保你的正常图片存放在类似 data/metal_nut/train/good/ 下
DATA_ROOT = 'data/grid/train' # 根据你实际解压的路径调整，比如 'factory_data/metal_nut/train'
train_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- 3. 模型初始化 ---
model = ConvAutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. 开始训练 ---
loss_history = []
print(f"正在 {device} 上开始训练金属螺母模型...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0  # 每一轮开始前清零
    
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
# 【修改 2】：保存的权重名字改为 metal_nut_ae.pth
torch.save(model.state_dict(), 'weights/grid_ae.pth')

# 【修改 3】：Loss 记录文件改名，避免覆盖之前瓶子 (bottle) 的训练数据
with open('results/loss_grid.txt', 'w') as f:
    for l in loss_history:
        f.write(f"{l}\n")

print("训练完成！网格模型已保存至 weights/grid_ae.pth，Loss 数据已存入 loss_grid.txt")