import streamlit as st
import pandas as pd
import sqlite3
import os

# ==========================================
# 0. 全局页面配置
# ==========================================
st.set_page_config(page_title="数据中心", page_icon="📊", layout="wide")

# ==========================================
# 1. 侧边栏配置 (统一 GitHub 入口)
# ==========================================
st.sidebar.markdown("### 📊 报表控制台")
st.sidebar.info("当前模块：**历史质检追溯系统**")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 源码与资源")
st.sidebar.markdown(
    """
    [📁 GitHub: Defect-Detection-CAE](https://github.com/Changes7/Defect-Detection-CAE)
    
    **版本**: v1.1.0 (Stable)
    """
)
st.sidebar.markdown(
    "[![GitHub stars](https://img.shields.io/github/stars/Changes7/Defect-Detection-CAE?style=social)](https://github.com/Changes7/Defect-Detection-CAE)"
)

st.title("📊 历史质检数据中心")

db_path = "data/industrial_inspection.db"

# ==========================================
# 2. 高效数据加载 (带 5 秒 TTL 缓存)
# ==========================================
@st.cache_data(ttl=5)
def load_historical_data(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    
    conn = sqlite3.connect(path)
    try:
        # 一次性读取并转换时间戳
        df = pd.read_sql_query("SELECT * FROM inspection_logs", conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

# ==========================================
# 3. 页面渲染逻辑
# ==========================================
df = load_historical_data(db_path)

if df.empty:
    st.warning("⚠️ 暂无数据库记录。请前往【实时缺陷检测】模块生成测试数据。")
else:
    # --- 第一部分：完整明细表 (用户要求放在第一) ---
    st.subheader("📁 完整检测日志检索")
    st.caption("💡 提示：点击表头可进行升/降序排列，鼠标悬停右上角可导出数据。")
    
    # 按照时间倒序显示，让最新的记录排在最前面
    display_df = df.sort_values('timestamp', ascending=False)
    st.dataframe(
        display_df[['timestamp', 'product_type', 'result', 'defect_count', 'mse']], 
        use_container_width=True,
        height=350 
    )

    st.markdown("---")

    # --- 第二部分：可视化看板 ---
    st.subheader("📈 数据可视化看板")
    
    tab1, tab2 = st.tabs(["📉 生产线残差趋势", "📊 质量分布统计"])
    
    with tab1:
        st.markdown("**CAE 模型重构误差 (MSE) 时间序列**")
        # 趋势图需要时间正序
        trend_data = df.sort_values('timestamp').set_index('timestamp')['mse']
        st.line_chart(trend_data)
        st.caption("注：MSE 峰值代表模型识别到的结构性瑕疵程度。")
        
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**合格率分布 (OK vs NG)**")
            counts = df['result'].value_counts()
            st.bar_chart(counts)
            
        with col_b:
            st.markdown("**产品类型质检频次**")
            p_counts = df['product_type'].value_counts()
            st.bar_chart(p_counts)

# 底部版权
st.markdown("---")
st.caption("工业产品表面缺陷检测系统 · 历史报表模块 · 2026")