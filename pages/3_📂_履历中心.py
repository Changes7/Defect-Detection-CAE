import streamlit as st
import pandas as pd
import sqlite3
from core_utils import get_all_data, DB_NAME
from style_utils import apply_glass_theme
apply_glass_theme()

st.title("📂 质检履历管理数据库")

df = get_all_data()

if not df.empty:
    # 顶部统计与操作区
    col1, col2 = st.columns([3, 1])
    with col1:
        # 筛选器
        target = st.multiselect("🔍 按产品筛选", df['product_type'].unique(), default=df['product_type'].unique())
        filtered_df = df[df['product_type'].isin(target)]
    
    with col2:
        # 导出功能 (utf-8-sig 防止 Excel 打开中文乱码)
        st.markdown("<br>", unsafe_allow_html=True)
        csv = filtered_df.to_csv(index=False).encode('utf-8-sig') 
        st.download_button(
            label="📥 导出报表 (CSV)",
            data=csv,
            file_name="inspection_report.csv",
            mime="text/csv",
            use_container_width=True
        )

    # 动态统计卡片
    st.markdown("### 📊 当前筛选数据概览")
    f_total = len(filtered_df)
    f_ng = len(filtered_df[filtered_df['result'].str.contains('不合格')])
    f_ok = f_total - f_ng
    
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("筛选记录数", f"{f_total} 条")
    cc2.metric("合格品数", f"{f_ok} 件")
    cc3.metric("当前条件良品率", f"{(f_ok/f_total)*100:.1f}%" if f_total > 0 else "0%")
    
    # 数据表展示
    st.dataframe(filtered_df, use_container_width=True)
    
    # 清理功能区
    st.markdown("---")
    with st.expander("⚠️ 危险操作区 (管理员数据清理)"):
        st.warning("点击下方按钮将清空所有历史质检数据（用于演示前清理测试脏数据）。此操作不可逆！")
        if st.button("🗑️ 清空所有测试数据"):
            conn = sqlite3.connect(DB_NAME)
            conn.execute("DROP TABLE IF EXISTS detection_logs")
            conn.commit()
            conn.close()
            st.cache_data.clear() # 清除缓存
            st.success("数据已全部清空！请刷新页面。")
            st.rerun()

else:
    st.info("💡 数据库暂无记录，请先前往【AI 质检终端】进行测试。")