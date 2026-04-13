import streamlit as st
import pandas as pd
import sqlite3
import os

# ==========================================
# 0. 全局页面配置
# ==========================================
st.set_page_config(page_title="数据报表", page_icon="📊", layout="wide")

st.title("📊 历史质检数据中心")

db_path = "data/industrial_inspection.db"

# ==========================================
# 1. 数据库读取逻辑
# ==========================================
if not os.path.exists(db_path):
    st.warning("暂无数据库记录，请先进行缺陷检测。")
else:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT * FROM inspection_logs", conn)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp', ascending=False) # 默认按时间倒序
    except Exception as e:
        st.error(f"读取失败: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    if not df.empty:
        # --- 第一部分：关键明细表 (用户要求放在第一) ---
        st.subheader("📁 完整检测日志检索")
        # 增加搜索和筛选的简洁提示
        st.caption("提示：点击表头可排序，鼠标悬停右上角可下载 CSV")
        st.dataframe(
            df[['timestamp', 'product_type', 'result', 'defect_count', 'mse']], 
            use_container_width=True,
            height=300 # 固定高度，防止表格太长淹没下方的图表
        )

        st.markdown("---")

        # --- 第二部分：可视化分析 (采用选项卡，保持界面整洁) ---
        st.subheader("📈 数据可视化看板")
        
        tab1, tab2 = st.tabs(["📉 残差趋势分析", "📊 质量分布统计"])
        
        with tab1:
            st.markdown("**生产线残差 (MSE) 波动**")
            # 画趋势图需要按时间正序
            trend_data = df.sort_values('timestamp').set_index('timestamp')['mse']
            st.line_chart(trend_data)
            
        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**合格率分布 (OK vs NG)**")
                st.bar_chart(df['result'].value_counts())
            with col_b:
                st.markdown("**产品类型检测频次**")
                st.bar_chart(df['product_type'].value_counts())

    else:
        st.info("数据库目前为空。")

# --- 侧边栏清理 ---
st.sidebar.info("数据报表模块：实时同步 SQLite 数据库记录，支持多维度历史追溯。")