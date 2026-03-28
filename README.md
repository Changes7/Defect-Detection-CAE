# 🤖 工业产品表面缺陷检测系统 (Industrial Defect Detection System)

本项目是一款基于 **卷积自编码器 (Convolutional Autoencoder, CAE)** 的无监督深度学习系统，旨在实现对工业产品（如瓶口、金属零件）表面瑕疵的自动化精准定位。

## 🌟 项目亮点 (Highlights)
- **改进架构**：引入 **Batch Normalization (BN)** 层，显著加速模型收敛并提升重构精度。
- **智能定位**：基于重构残差分析，结合 OpenCV 实现瑕疵区域的自动边界框 (Bounding Box) 标注。
- **交互界面**：集成 **Streamlit Web App**，支持用户上传图片并实时查看检测报告。

## 📁 目录结构
- `data/`: 原始图像数据集。
- `weights/`: 存储训练好的模型权重 (`.pth`)。
- `results/`: 存储训练 Loss 曲线及检测结果图。
- `app.py`: **[新]** Streamlit Web 交互式 Demo 脚本。
- `model.py`: CAE 神经网络架构定义。
- `train.py` / `test.py`: 核心训练与单机测试脚本。

## 📊 实验成果展示

### 1. 训练收敛曲线
模型在引入 BN 层后表现出极佳的稳定性：
![Loss Curve](./results/loss_curve_result.png)

### 2. Web 交互界面预览
通过网页端可一键实现“上传-重构-定位”全流程：
![Web Interface](./results/defect_detection_final.png)
*(注：此处可替换为您部署成功后的网页截图)*

## 🚀 如何运行
1. **安装环境**：`pip install -r requirements.txt`
2. **启动本地 Web Demo**：
   ```bash
   streamlit run app.py
   