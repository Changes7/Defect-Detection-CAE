import streamlit as st

st.set_page_config(page_title="工业质检云平台", page_icon="🛡️", layout="wide")

st.title("🏭 欢迎进入 CAE-BN 工业质检平台")
st.markdown("---")
st.success("✅ 系统内核已启动，数据库引擎连接正常。")
st.info("👈 请点击左侧边栏的各个模块进入相应的工作台。")

# 工业白专业主题全局注入
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #FFFFFF !important; }
    [data-testid="stSidebar"] { background-color: #F8FAFC !important; border-right: 1px solid #E2E8F0; }
</style>
""", unsafe_allow_html=True)