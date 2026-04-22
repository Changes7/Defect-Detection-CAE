# 🚀 基于卷积自编码器(CAE)的工业产品表面缺陷检测系统
**Industrial Surface Defect Detection System based on CAE**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌐 在线演示 (Cloud Demo)
您可以直接访问部署在腾讯云的演示平台，实时体验 AI 质检过程：
👉 **[点击进入：数字化质检看板](http://152.136.23.81:8501)**
*(注：云端算力有限，推理耗时约 200ms-500ms)*

---

## 📖 项目简介 (Project Overview)
本项目是针对工业生产线中产品表面缺陷（如划痕、裂纹、异物等）设计的智能质检方案。采用 **卷积自编码器 (Convolutional Autoencoder)** 架构，通过无监督学习正常样本的特征空间，实现对异常区域的精准定位与判定。

### 核心功能：
- **实时推理**：秒级上传，实时反馈检测结果。
- **热力图可视化**：直观展示图像重构误差分布。
- **微服务架构**：FastAPI 提供高性能 AI 接口，Streamlit 提供交互式数据看板。
- **高可用部署**：基于 PM2 进程守护，实现 24/7 在线服务。

---

## 🏗️ 系统架构 (Architecture)


```mermaid
graph LR
    A[工业摄像机/用户上传] --> B(Streamlit 前端大屏)
    B -->|RESTful API| C(FastAPI 后端大脑)
    C -->|CAE Inference| D[PyTorch 推理引擎]
    D -->|检测结果/评分| B