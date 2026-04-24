#!/bin/bash
echo "正在通过 PM2 启动工业质检系统..."

# 停止旧进程（如果存在）
pm2 stop all

# 启动后端与前端
pm2 start api_server.py --name "cae-backend" --interpreter python3
pm2 start "streamlit run app.py --server.port 8501" --name "cae-frontend"

# 显示结果
pm2 list
echo "系统已点火，请访问 http://你的公网IP:8501"