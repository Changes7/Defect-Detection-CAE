import torch
import os
from PIL import Image
import matplotlib.pyplot as plt

# 1. 验证环境
print("--- 环境检查 ---")
print(f"PyTorch 版本: {torch.__version__}")

# 2. 检查数据路径 (假设你之前下载了解压在 data 文件夹)
data_dir = "./data/bottle/train/good"

if os.path.exists(data_dir):
    files = os.listdir(data_dir)
    print(f"成功！在训练集里找到了 {len(files)} 张正常瓶子的照片。")
    
    # 读取第一张图
    img_path = os.path.join(data_dir, files[0])
    img = Image.open(img_path)
    
    # 绘图并保存（远程连接无法直接弹窗，我们保存为图片看）
    plt.imshow(img)
    plt.title("Normal Bottle Sample")
    plt.axis('off')
    plt.savefig("sample_check.png")
    print("已生成预览图: sample_check.png，请在左侧文件栏双击查看。")
else:
    print(f"报错：找不到路径 {data_dir}。请确认 data 文件夹是否已下载解压。")