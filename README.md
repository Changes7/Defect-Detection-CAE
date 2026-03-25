# 工业产品表面缺陷检测系统 (Industrial Defect Detection CAE)

本项目是基于 **卷积自编码器 (Convolutional Autoencoder, CAE)** 实现的无监督缺陷检测系统，专门针对瓶口、金属螺母等工业产品的表面瑕疵。

## 1. 项目结构 (Directory Structure)
本项目遵循工程化开发规范，确保代码与数据的严格分离：
- `data/`: 原始图像数据集。
- `weights/`: 存储训练完成的模型权重文件 (`.pth`)。
- `results/`: 存储 Loss 曲线、残差热力图及最终检测结果图。
- `model.py`: CAE 神经网络架构定义（含 BatchNorm 层）。
- `train.py` / `test.py`: 核心训练与自动化检测脚本。

## 2. 核心改进：引入 Batch Normalization
为了解决深层网络训练难、收敛慢的问题，本项目在卷积层后引入了 **批归一化 (BN)** 层：
- **加速收敛**：实验证明，引入 BN 后模型在前 5 个 Epoch 即可实现 Loss 的快速下降。
- **训练稳定性**：有效缓解了梯度消失问题，使重构图像的细节更加清晰。

## 3. 实验结果展示 (Experimental Results)

### 3.1 训练收敛曲线
下图展示了模型在 50 轮训练中的 MSE 损失下降情况。可以看到曲线平滑且收敛迅速：
![训练损失曲线](./results/loss_curve_result.png)

### 3.2 自动化瑕疵定位效果
利用重构残差分析与 OpenCV 形态学处理，系统可自动圈定瑕疵区域：
![自动化缺陷定位](./results/defect_detection_final.png)
*注：系统自动识别并利用红色边界框（Bounding Box）标出了瓶口的破损位置。*

## 4. 如何运行 (Usage)
1. **准备环境**：`pip install -r requirements.txt`
2. **启动训练**：`python train.py`
3. **执行检测**：`python test.py` (结果将自动保存至 `results/` 目录)