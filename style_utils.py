# style_utils.py
import streamlit as st

def apply_glass_theme():
    st.markdown("""<style>
    /* 1. 侧边栏背景：降低不透明度，增加模糊深度 */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(25px) saturate(180%); /* 增加饱和度，让玻璃不发灰 */
        -webkit-backdrop-filter: blur(25px) saturate(180%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* 2. 导航容器：增加顶部间距，显得更有呼吸感 */
    [data-testid="stSidebarNav"] {
        padding-top: 0px !important;
        background-image: none !important; /* 移除 Streamlit 默认的顶部 Logo 线 */
    }

    /* 3. 功能键：去掉生硬的边框，改用微弱的阴影 */
    [data-testid="stSidebarNav"] ul li a {
        margin: 4px 16px !important;
        padding: 10px 16px !important;
        border-radius: 16px !important; /* 更圆润的 Gemini 风格 */
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1) !important; /* 关键：丝滑动画曲线 */
        color: rgba(255, 255, 255, 0.7) !important;
        position: relative;
        overflow: hidden;
    }

    /* 4. 悬停效果：不再是简单的变色，而是“微光”亮起 */
    [data-testid="stSidebarNav"] ul li a:hover {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        transform: scale(1.02) translateX(4px); /* 极其轻微的放大和右移 */
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }

    /* 5. 选中状态：放弃生硬的粗线，改用胶囊高亮 */
    [data-testid="stSidebarNav"] ul li a[data-active="true"] {
        background: rgba(59, 130, 246, 0.15) !important; /* 柔和的蓝色底色 */
        color: #60a5fa !important;
        font-weight: 600 !important;
        box-shadow: inset 0 0 10px rgba(59, 130, 246, 0.1); /* 内部微光 */
    }
                
    /* 6. 隐藏不必要的元素，保持极简 */
    section[data-testid="stSidebar"] [data-testid="stSidebarNavSeparator"] {
        display: none;
    }
                /* 给全局背景加一点点灰度，用来衬托纯白的卡片 */
    .stApp {
        background-color: #f8fafc; 
    }

    /* 指标卡片化 */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px; /* 增加内边距让呼吸感更强 */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); /* 柔和的阴影 */
        transition: transform 0.2s ease; /* 鼠标悬停微动效准备 */
    }
    
    /* 可选：鼠标滑过卡片微微上浮 */
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)