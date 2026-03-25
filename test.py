import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import ConvAutoEncoder
import os

# --- 1. 环境与模型准备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoEncoder().to(device)
model.load_state_dict(torch.load('metal_nut_ae.pth')) # 加载你刚练好的大脑
model.eval() # 开启测试模式

# --- 2. 准备一张“坏瓶子”的照片 ---
# 我们从测试集里找一个带划痕（broken）的瓶子
test_img_path = './data/metal_nut/test/scratch/000.png' 

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# 读取并预处理
img = Image.open(test_img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

# --- 3. AI 进行“还原” ---
with torch.no_grad():
    reconstructed = model(img_tensor)

# --- 4. 计算差异（这就是寻找缺陷的过程） ---
# 将 Tensor 转回 numpy 图片格式方便显示
orig_np = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
recon_np = reconstructed.squeeze().cpu().numpy().transpose(1, 2, 0)

# 计算“残差图”：原图和还原图的绝对差值
# 哪里差值大，哪里就是瑕疵！
diff = np.abs(orig_np - recon_np)
diff_gray = np.mean(diff, axis=2) # 转为灰度图看缺陷更明显

# --- 5. 结果可视化 ---
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original (Defective)")
plt.imshow(orig_np)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("AI Reconstruction (As 'Good')")
plt.imshow(recon_np)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Anomaly Map (The Defect!)")
plt.imshow(diff_gray, cmap='jet') # 用彩色图显示，红色的地方就是瑕疵
plt.colorbar()
plt.axis('off')

plt.savefig('result_comparison.png')
print("检测完成！请查看生成的结果图：result_comparison.png")