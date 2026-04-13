import streamlit as st
import pandas as pd
import sqlite3
import os
import datetime
from streamlit_autorefresh import st_autorefresh # 新增：引入自动刷新组件

# ==========================================
# 0. 全局页面配置
# ==========================================
st.set_page_config(page_title="工业质检平台", page_icon="🏭", layout="wide")

# ==========================================
# 💥 1. 核心大招：全自动轮询刷新引擎
# ==========================================
# 每隔 5000 毫秒（5秒）自动刷新一次整个页面，实现 Live 实时看板
st_autorefresh(interval=5000, limit=10000, key="data_refresh")

# ==========================================
# 2. 页面标题与动态状态栏
# ==========================================
st.title("🏭 CAE-BN 工业产品表面智能质检平台")
# 把当前时间写进 caption，你会看到网页上的秒数每过 5 秒就会自动跳变！
st.caption(f"🟢 系统处于 Live 模式 | 看板数据每 5 秒自动同步底层数据库 | 当前系统时间：**{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")
st.markdown("---")

# ==========================================
# 3. 核心逻辑：执行 SQL 查询获取真实指标
# ==========================================
def get_realtime_metrics():
    db_path = "data/industrial_inspection.db"
    
    # 如果数据库文件还不存在
    if not os.path.exists(db_path):
        return 0, 100.0, 0, 0
    
    conn = sqlite3.connect(db_path)
    try:
        # 利用 Pandas 直接执行 SQL 查询
        df = pd.read_sql_query("SELECT * FROM inspection_logs", conn)
    except sqlite3.OperationalError:
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
# 4. 渲染仪表盘 (Dashboard)
# ==========================================
st.subheader("📋 生产线实时看板 (Live Data)")
col1, col2, col3, col4 = st.columns(4)

col1.metric(label="今日已检测总数", value=f"{total} 件")
col2.metric(label="实时良品率", value=f"{y_rate:.1f}%")
col3.metric(label="发现缺陷件数", value=f"{ng} 件", delta="待复检", delta_color="inverse")
col4.metric(label="合格放行件数", value=f"{ok} 件")

st.markdown("<br>", unsafe_allow_html=True)

# ==========================================
# 5. 从数据库中拉取最新流水
# ==========================================
if total > 0:
    st.write("🕒 **最新检测流水**")
    conn = sqlite3.connect("data/industrial_inspection.db")
    # 高效拉取最新 5 条记录
    query = """
        SELECT timestamp as 时间, 
               product_type as 产品类型, 
               result as 检测结果, 
               defect_count as 缺陷数量, 
               mse as 均方误差 
        FROM inspection_logs 
        ORDER BY id DESC 
        LIMIT 5
    """
    df_recent = pd.read_sql_query(query, conn)
    conn.close()
    st.table(df_recent)
else:
    st.info("💡 目前尚无检测记录。请前往左侧【实时缺陷检测】开始工作。")

st.markdown("---")
st.info("**系统架构说明**：本首页数据已接入底层 SQLite 关系型数据库，并启用了 Web 端的全自动轮询（5秒/次），实现了高并发场景下真正的物联网（IoT）级实时监控。")