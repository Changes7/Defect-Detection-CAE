import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2  # 用于图像处理和画框 (需要虚拟机里装了 opencv-python)
from model import ConvAutoEncoder
import os

# 设置 Matplotlib 解决中文显示问题（如果虚拟机没有中文字体，请保持默认用英文）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 

# --- 1. 配置参数 ---
MODEL_PATH = 'bottle_ae.pth'               # 训练好的模型路径
TEST_IMAGE = 'data/bottle/test/broken_small/001.png' # 选择一张缺陷图片进行测试 (请确保文件存在)
SAVE_RESULT = 'defect_detection_final.png'   # 最终结果保存路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"正在 {device} 上运行测试...")

# --- 2. 加载模型与权重 ---
model = ConvAutoEncoder().to(device)
if os.path.exists(MODEL_PATH):
    # map_location 确保在不同设备上也能加载
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"成功加载模型：{MODEL_PATH}")
else:
    print(f"错误：找不到模型文件 {MODEL_PATH}")
    exit()

# --- 3. 图像预处理 ---
# 必须和训练时的 preprocess 完全一致
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), # 归一化到 0-1
])

# 加载测试图片并转换
def load_and_transform_image(image_path):
    if not os.path.exists(image_path):
        print(f"错误：找不到测试图片 {image_path}")
        exit()
    img_pil = Image.open(image_path).convert('RGB')
    # 保留原始图片尺寸的 numpy 格式，专门用于 OpenCV 画红框
    origin_img_np = np.array(img_pil) 
    # 转换为模型输入的 Tensor (1, 3, 256, 256)
    img_tensor = transform(img_pil).unsqueeze(0).to(device) 
    return img_tensor, origin_img_np

test_tensor, origin_img_np = load_and_transform_image(TEST_IMAGE)

# --- 4. 前向传播：获取重构图 ---
with torch.no_grad():
    recon_tensor = model(test_tensor)

# --- 5. 计算残差热力图 (Heatmap) ---
# 计算像素级 MSE (Original - Reconstructed)
# 计算 RGB 三个通道的平均 MSE
error_map_tensor = torch.mean(torch.pow(test_tensor - recon_tensor, 2), dim=1).squeeze().cpu()
error_map_np = error_map_tensor.numpy()

# 归一化误差图到 0-255，方便 OpenCV 进行阈值处理
heatmap_norm = (error_map_np * 255).astype(np.uint8)

# --- 6. 自动化缺陷检测与画框 (核心改进部分) ---
# 6.1 设定自动化阈值
# 这是一个非常实用的毕设方案：取所有像素误差的第 99.5 百分位作为动态阈值
# 大于此值的被认定为瑕疵点
threshold_value = np.percentile(error_map_np, 99.5) * 255
_, mask = cv2.threshold(heatmap_norm, threshold_value, 255, cv2.THRESH_BINARY)

# 6.2 进行形态学“闭运算” (先膨胀再腐蚀)，去除微小噪声点，让分散的瑕疵连成片
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# 6.3 准备原图副本用于画框 (调整尺寸为 256x256 匹配模型)
# 因为 OpenCV 使用 BGR 格式，需要做一下转换
img_for_bbox = cv2.resize(origin_img_np, (256, 256))
img_for_bbox_bgr = cv2.cvtColor(img_for_bbox, cv2.COLOR_RGB2BGR)

# 6.4 寻找连通域并绘制边界框 (Bounding Boxes)
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

defect_count = 0
for cnt in contours:
    # 过滤掉面积太小的噪声块
    if cv2.contourArea(cnt) > 20: 
        x, y, w, h = cv2.boundingRect(cnt)
        # 绘制红色的框 (BGR 格式: 0, 0, 255)，线条宽度为 2
        cv2.rectangle(img_for_bbox_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
        defect_count += 1

# 转换回 RGB 用于 Matplotlib 显示
img_with_boxes_rgb = cv2.cvtColor(img_for_bbox_bgr, cv2.COLOR_BGR2RGB)

print(f"检测完成，在图片中找到 {defect_count} 个可能的瑕疵区域。")

# --- 7. 结果可视化与保存 (专业的 2x2 排版) ---
# 将 Tensor 转换回可显示的 numpy 格式 (256, 256, 3)
def tensor_to_np(t):
    return t.squeeze().permute(1, 2, 0).cpu().numpy()

img_in_np = tensor_to_np(test_tensor)
recon_in_np = tensor_to_np(recon_tensor)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 7.1 原始图片
axes[0, 0].imshow(img_in_np)
axes[0, 0].set_title('Original Bad Sample')
axes[0, 0].axis('off')

# 7.2 模型重构图 (AI 认为它应该是“完美”的样子)
axes[0, 1].imshow(recon_in_np)
axes[0, 1].set_title('AI Reconstruction')
axes[0, 1].axis('off')

# 7.3 残差热力图 (差异越大，颜色越深/暖)
# 使用 'jet' 映射，把误差变成红黄蓝热力图
axes[1, 0].imshow(heatmap_norm, cmap='jet')
axes[1, 0].set_title('Reconstruction Residual (MSE)')
axes[1, 0].axis('off')

# 7.4 自动化画框检测结果 (最直观的成果展示)
axes[1, 1].imshow(img_with_boxes_rgb)
axes[1, 1].set_title(f'Automated Detection (Found {defect_count})')
axes[1, 1].axis('off')

# 保存高清结果图
plt.tight_layout()
plt.savefig(SAVE_RESULT, dpi=300, bbox_inches='tight')
print(f"结果已保存：{SAVE_RESULT}")