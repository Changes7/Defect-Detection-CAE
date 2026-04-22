# 🏭 CAE-BN: 工业产品表面缺陷智能质检平台

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-orange)

## 🌟 项目简介
本项目是一款基于**卷积自编码器 (CAE)** 的无监督工业缺陷检测系统。采用**微服务架构**设计，将 AI 推理引擎与 Web 交互终端彻底解耦，适用于 AGV 智能仓储、流水线实时质检等工业 4.0 场景。

点击查看：[我的在线演示地址](http://192.168.204.133:8501)

# 演示与文档
[![Demo](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge&logo=streamlit)](http://192.168.204.133:8501)
[![API Docs](https://img.shields.io/badge/API-Docs-green?style=for-the-badge&logo=fastapi)](http://192.168.204.133:8000/docs)

## 🛠️ 技术架构
- **核心算法**: 基于 PyTorch 的卷积自编码器 (Convolutional Autoencoder)
- **后端引擎**: FastAPI (RESTful API, 支持高并发推理)
- **前端界面**: Streamlit (响应式工业看板)
- **安全保障**: 基于 RBAC 的权限控制系统 + SHA-256 密码加密
- **部署方案**: 支持 Docker 容器化部署与一键 Shell 脚本运维



## 🚀 快速启动

### 1. 环境安装
```bash
git clone [https://github.com/你的用户名/defect_detection.git](https://github.com/你的用户名/defect_detection.git)
cd defect_detection
python -m venv venv
source venv/bin/activate  # Linux/Ubuntu
pip install -r requirements.txt
   