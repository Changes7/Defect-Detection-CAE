import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import ConvAutoEncoder
import os
from pytorch_msssim import ssim

# ==========================================
# 1. 核心配置（训练不同产品只需改这里）
# ==========================================
PRODUCT = "grid"  # 瓶底
DATA_ROOT = f'data/{PRODUCT}/train'
SAVE_PATH = f'weights/{PRODUCT}_ae.pth'
CHECKPOINT_PATH = f'weights/{PRODUCT}_checkpoint.pth' # 临时存档点
LOSS_LOG_PATH = f'results/loss_{PRODUCT}.txt'

EPOCHS = 200
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
ALPHA = 0.8  # 复合损失权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 安全检查与环境准备
# ==========================================
for path in ['weights', 'results', 'data']:
    os.makedirs(path, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 加载数据集
try:
    train_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
except FileNotFoundError:
    print(f"❌ 错误：找不到路径 {DATA_ROOT}，请检查文件夹是否解压正确！")
    exit()

# ==========================================
# 3. 模型与断点接力
# ==========================================
model = ConvAutoEncoder().to(device)
mse_criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

start_epoch = 0
# 【关键：接力逻辑】
if os.path.exists(CHECKPOINT_PATH):
    print(f"♻️ 发现【{PRODUCT}】的断点存档，正在恢复进度...")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print("✅ 进度恢复成功，将从上次中断的位置继续训练。")

# ==========================================
# 4. 开始训练（带防中断保护）
# ==========================================
loss_history = []
print(f"🚀 正在 {device} 上启动【{PRODUCT}】训练...")

try:
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0
        
        for img, _ in train_loader:
            img = img.to(device)
            output = model(img)
            
            # 计算复合 Loss (MSE + SSIM)
            loss_mse = mse_criterion(output, img)
            loss_ssim = 1 - ssim(output, img, data_range=1.0, size_average=True)
            loss = ALPHA * loss_mse + (1 - ALPHA) * loss_ssim
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)
        
        # 每 10 轮打印一次并存档
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], 综合 Loss: {avg_loss:.4f}')
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            # 同时也更新一下 Loss 日志，防止中断丢失
            with open(LOSS_LOG_PATH, 'a') as f:
                f.write(f"{avg_loss}\n")

except KeyboardInterrupt:
    # 【关键：紧急避险】手动按 Ctrl+C 时触发
    print("\n🛑 检测到中断！正在保存当前进度到存档点...")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"✅ 存档已保存至 {CHECKPOINT_PATH}。下次运行将自动接力！")
    exit()

# ==========================================
# 5. 训练圆满完成
# ==========================================
torch.save(model.state_dict(), SAVE_PATH)
# 训练完成后，删除临时存档点
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)

print(f"🎊 【{PRODUCT}】训练圆满完成！")
print(f"1. 权重：{SAVE_PATH}")
print(f"2. 日志：{LOSS_LOG_PATH}")