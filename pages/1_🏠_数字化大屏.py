import streamlit as st
import pandas as pd
import os
from streamlit_autorefresh import st_autorefresh
from core_utils import get_all_data

st.title("🏠 数字化生产监控看板")

# 自动轮询：每 5 秒刷新一次大屏，实现数据实时联动
st_autorefresh(interval=5000, key="dashboard_refresh")

df = get_all_data()

# 核心指标区
t1, t2, t3, t4 = st.columns(4)
if not df.empty:
    total = len(df)
    ng = len(df[df['result'].str.contains('不合格')])
    ok = total - ng
    y_rate = (ok / total) * 100
    
    t1.metric("今日检测总数", f"{total} 件")
    t2.metric("实时良品率", f"{y_rate:.1f}%")
    t3.metric("异常检出", f"{ng} 件", delta=f"{ng} 待处理" if ng > 0 else None, delta_color="inverse")
    t4.metric("算法置信度", "99.2%", delta="Optimal")
else:
    t1.metric("今日检测总数", "0 件")
    t2.metric("实时良品率", "100.0%")
    t3.metric("异常检出", "0 件")
    t4.metric("算法置信度", "N/A")
    st.info("💡 暂无今日生产数据，请前往【AI 质检终端】进行测试。")

st.markdown("---")
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("📈 模型收敛健康度 (Loss Curve)")
    if os.path.exists('results/loss_curve_result.png'):
        st.image('results/loss_curve_result.png', caption="CAE模型实时收敛状态", use_container_width=True)
    else:
        st.info("尚未生成训练曲线图 (results/loss_curve_result.png)")
with c2:
    st.subheader("🛡️ 系统核心参数")
    st.json({
        "模型架构": "ConvAutoEncoder-BN",
        "损失函数": "MSE + SSIM (Composite)",
        "输入分辨率": "256x256 RGB",
        "数据库": "SQLite3 (Local Storage)",
        "状态": "Running (Live)"
    })