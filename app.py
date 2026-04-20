import streamlit as st
import pandas as pd
import sqlite3
import os
import datetime
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 0. 全局页面配置 (必须是第一个 Streamlit 命令)
# ==========================================
st.set_page_config(page_title="工业质检平台", page_icon="🏭", layout="wide")

# ==========================================
# 1. 全自动轮询刷新引擎 (5秒/次)
# ==========================================
st_autorefresh(interval=5000, limit=10000, key="data_refresh")

# ==========================================
# 2. 侧边栏：系统信息与源码入口
# ==========================================
st.sidebar.title("🛠️ 系统控制台")
st.sidebar.info("当前系统运行于：**生产线实时监控模式**")

# 展示项目源码链接，彰显代码规范性
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 资源与源码")
st.sidebar.markdown(
    """
    本系统源码已同步至 GitHub 仓库。
    
    [📁 前往仓库：Defect-Detection-CAE](https://github.com/Changes7/Defect-Detection-CAE)
    
    **版本**: v1.1.0 (Stable)  
    **架构**: PyTorch + SQLite
    """
)
st.sidebar.markdown("[![GitHub stars](https://img.shields.io/github/stars/Changes7/Defect-Detection-CAE?style=social)](https://github.com/Changes7/Defect-Detection-CAE)")

# ==========================================
# 7. 侧边栏：检测报告下载功能
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 检测报告下载")

# 生成检测报告的函数
@st.cache_data(ttl=10)
def generate_inspection_report():
    db_path = "data/industrial_inspection.db"
    if not os.path.exists(db_path):
        return None
    
    conn = sqlite3.connect(db_path)
    try:
        # 获取所有检测记录，按时间降序排列
        query = """
            SELECT timestamp as 检测时间, 
                   product_type as 模型名称, 
                   result as 系统判定结果, 
                   mse as MSE误差值
            FROM inspection_logs 
            ORDER BY timestamp DESC
        """
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        st.sidebar.error(f"读取数据库时出错: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()
    
    return df

# 获取报告数据
report_df = generate_inspection_report()

if report_df is not None and not report_df.empty:
    # 将DataFrame转换为CSV
    csv_data = report_df.to_csv(index=False, encoding='utf-8-sig')
    
    # 生成当前时间戳作为文件名的一部分
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"缺陷检测报告_{current_time}.csv"
    
    # 创建下载按钮
    st.sidebar.download_button(
        label="📥 一键下载完整检测报告 (CSV)",
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
        help="下载包含所有检测记录的完整报告，包括检测时间、模型名称、MSE误差值和系统判定结果"
    )
    
    # 显示报告摘要
    st.sidebar.info(f"报告包含 **{len(report_df)}** 条检测记录，最新记录：{report_df.iloc[0]['检测时间']}")
elif report_df is None:
    st.sidebar.warning("数据库不存在，无法生成报告。")
else:
    st.sidebar.warning("暂无检测记录，无法生成报告。")

# ==========================================
# 3. 页面标题与动态状态栏
# ==========================================
st.title("🏭 CAE-BN 工业产品表面智能质检平台")
st.caption(f"🟢 系统处于 Live 模式 | 自动同步数据库 | 当前时间：**{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
st.markdown("---")

# ==========================================
# 4. 核心逻辑：高效读取数据库 (使用新版缓存)
# ==========================================
# 这里使用 st.cache_data 代替已废弃的 experimental_memo，确保性能且不报错
@st.cache_data(ttl=5) # 缓存5秒，配合自动刷新，既省电又实时
def get_realtime_metrics():
    db_path = "data/industrial_inspection.db"
    if not os.path.exists(db_path):
        return 0, 100.0, 0, 0
    
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query("SELECT result FROM inspection_logs", conn)
    except Exception:
        conn.close()
        return 0, 100.0, 0, 0
    conn.close()
    
    total_count = len(df)
    if total_count == 0:
        return 0, 100.0, 0, 0
    
    ng_count = len(df[df['result'] == 'NG'])
    ok_count = total_count - ng_count
    yield_rate = (ok_count / total_count) * 100
    return total_count, yield_rate, ng_count, ok_count

total, y_rate, ng, ok = get_realtime_metrics()

# ==========================================
# 5. 渲染仪表盘 (Dashboard)
# ==========================================
st.subheader("📋 生产线实时看板 (Live Data)")
col1, col2, col3, col4 = st.columns(4)

col1.metric(label="今日已检测总数", value=f"{total} 件")
col2.metric(label="实时良品率", value=f"{y_rate:.1f}%")
col3.metric(label="发现缺陷件数", value=f"{ng} 件", delta="🚨 待复检" if ng > 0 else None, delta_color="inverse")
col4.metric(label="合格放行件数", value=f"{ok} 件")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 6. 从数据库中拉取最新流水
# ==========================================
if total > 0:
    st.write("🕒 **最新检测流水 (Top 5)**")
    conn = sqlite3.connect("data/industrial_inspection.db")
    query = """
        SELECT timestamp as 时间, 
               product_type as 产品类型, 
               result as 检测结果, 
               mse as 均方误差 
        FROM inspection_logs 
        ORDER BY id DESC 
        LIMIT 5
    """
    df_recent = pd.read_sql_query(query, conn)
    conn.close()
    st.table(df_recent)
else:
    st.info("💡 目前尚无检测记录。请前往左侧【智能检测终端】开始工作。")

st.markdown("---")
st.info("**系统架构说明**：本系统已成功解决旧版 API 兼容性问题，目前采用最新的 `st.cache_data` 技术保障数据流的高效稳定。")
