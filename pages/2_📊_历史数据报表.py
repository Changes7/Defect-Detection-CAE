import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="数据报表中心", page_icon="📊", layout="wide")

# --- 1. 数据库读取函数 ---
def get_data():
    db_path = "data/industrial_inspection.db"
    if not os.path.exists(db_path):
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    # 直接用 pandas 读取 SQL 结果，非常高效
    df = pd.read_sql_query("SELECT * FROM inspection_logs", conn)
    conn.close()
    return df

st.title("📊 生产线质检历史数据看板")

df = get_data()

if df.empty:
    st.warning("目前数据库中尚无检测记录，请先去【实时检测】页面进行测试。")
else:
    # --- 2. 顶部核心指标 (Metrics) ---
    total_count = len(df)
    ok_count = len(df[df['result'] == 'OK'])
    ng_count = len(df[df['result'] == 'NG'])
    ok_rate = (ok_count / total_count) * 100 if total_count > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("累计检测总数", f"{total_count} 件")
    m2.metric("实时良品率", f"{ok_rate:.1f}%")
    m3.metric("发现缺陷件数", f"{ng_count} 件", delta_color="inverse")
    m4.metric("合格放行件数", f"{ok_count} 件")

    st.markdown("---")

    # --- 3. 图表分析层 ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("✅ 合格比例分布")
        # 饼图：展示 OK 和 NG 的占比
        fig_pie = px.pie(df, names='result', color='result',
                         color_discrete_map={'OK':'#2ecc71', 'NG':'#e74c3c'},
                         hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("📦 各类产品检测统计")
        # 柱状图：展示不同 product_type 的检测分布
        prod_counts = df.groupby(['product_type', 'result']).size().reset_index(name='counts')
        fig_bar = px.bar(prod_counts, x='product_type', y='counts', color='result',
                         barmode='group',
                         color_discrete_map={'OK':'#2ecc71', 'NG':'#e74c3c'},
                         labels={'counts':'检测数量', 'product_type':'产品类别'})
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- 4. 原始数据穿透查看 ---
    with st.expander("🔍 查看原始明细数据"):
        # 允许用户按产品筛选
        selected_prod = st.multiselect("筛选产品类别", df['product_type'].unique(), default=df['product_type'].unique())
        filtered_df = df[df['product_type'].isin(selected_prod)]
        st.dataframe(filtered_df.sort_values(by='id', ascending=False), use_container_width=True)
        
        # 提供 CSV 下载功能
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 导出报表为 CSV", data=csv, file_name="inspection_report.csv", mime="text/csv")