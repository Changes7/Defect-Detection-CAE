#!/bin/bash
# 一键清理旧进程并重启系统

echo "正在停止旧服务..."
pm2 delete all

echo "正在启动后端推理引擎..."
pm2 start "python3 api_server.py" --name "ai-backend"

echo "正在启动前端可视化看板..."
pm2 start "streamlit run app.py --server.port 8501 --server.address 0.0.0.0" --name "ai-frontend"

echo "系统已上线！请访问 http://152.136.23.81:8501"
pm2 list