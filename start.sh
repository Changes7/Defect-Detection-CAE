#!/bin/bash

echo "====================================="
echo "🚀 CAE-BN 工业智能质检系统启动程序"
echo "====================================="

# 1. 激活虚拟环境 (确保路径正确)
source venv/bin/activate

# 2. 启动后台 AI 引擎 (末尾的 & 表示让它在后台静默运行)
echo "🧠 [1/2] 正在唤醒底层 AI 视觉引擎 (端口 8000)..."
python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 > api_log.txt 2>&1 &

# 等待 3 秒，确保引擎完全启动，避免前端连不上
sleep 3

# 3. 启动前端 Web 界面
echo "🖥️ [2/2] 正在加载质检 UI 大屏 (端口 8501)..."
streamlit run 🏠_应用.py

# 退出提示 (当你在终端按下 Ctrl+C 关掉网页时，顺手把后台的 AI 引擎也杀掉)
trap "echo '🛑 正在关闭系统...'; pkill -f uvicorn; exit" INT TERM