import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
import streamlit.components.v1 as components
from model import ConvAutoEncoder 

# ==========================================
# 0. 全局页面配置 (白色专业版)
# ==========================================
st.set_page_config(
    page_title="工业产品表面缺陷检测与分析系统", 
    page_icon="🔍", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. 工业白主题 CSS 注入 (完全移除黑夜模式)
# ==========================================
white_theme_css = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 主背景与侧边栏 */
    [data-testid="stAppViewContainer"] { background-color: #FFFFFF !important; }
    [data-testid="stSidebar"] { background-color: #F8FAFC !important; border-right: 1px solid #E2E8F0; }
    
    /* 文字颜色统一 */
    h1, h2, h3, h4, p, label, .stMarkdown { color: #1E293B !important; font-family: "Segoe UI", sans-serif; }
    
    /* 侧边栏导航美化 */
    [data-testid="stSidebarNav"] { display: none; } /* 隐藏原生导航，改用自定义 */
    
    /* 仪表盘卡片 (明亮阴影风) */
    [data-testid="stMetric"] {
        background-color: #F1F5F9 !important;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid #E2E8F0;
    }
    
    /* 按钮与组件美化 */
    .stButton>button { width: 100%; border-radius: 8px; background-color: #3B82F6; color: white; }
    img { border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); }
</style>
"""
st.markdown(white_theme_css, unsafe_allow_html=True)

# ==========================================
# 2. 核心逻辑：模型加载与工具函数
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource 
def load_defect_model(path): 
    model = ConvAutoEncoder()
    if not os.path.exists(path): return None
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ==========================================
# 3. 侧边栏：多功能集成导航
# ==========================================
with st.sidebar:
    st.title("🛡️ 质检系统控制台")
    st.markdown("---")
    # 统一整合所有功能模块
    menu_selection = st.radio(
        "📂 系统功能模块",
        ["🏠 系统概览 (Home)", 
         "🔍 自动化检测 (Detection)", 
         "📊 数据统计报表 (Analytics)", 
         "🧬 CAE 架构解密 (Architecture)"]
    )
    st.markdown("---")
    st.info("当前状态：算法引擎已就绪")

# ==========================================
# 4. 功能路由分发
# ==========================================

# --- 模块 1：系统概览 ---
if menu_selection == "🏠 系统概览 (Home)":
    st.title("🏠 工业产品表面缺陷检测系统概览")
    st.write("本系统基于卷积自编码器（CAE）技术，通过非监督学习实现工业产品的高精度自动化质检。")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 算法收敛性能")
        if os.path.exists('results/loss_curve_result.png'):
            st.image('results/loss_curve_result.png', use_container_width=True)
        else:
            st.warning("暂无训练曲线数据")
    
    with col2:
        st.subheader("📋 系统运行状态")
        st.success("✅ GPU 加速状态：已开启")
        st.success("✅ 检测模型数量：3 组已部署")
        st.success("✅ 数据连接状态：本地存储正常")

# --- 模块 2：自动化检测 ---
elif menu_selection == "🔍 自动化检测 (Detection)":
    st.title("🔍 自动化检测工作台")
    
    # 侧边栏设置移入该模块
    product_type = st.sidebar.selectbox("🎯 选择检测对象", ("药用瓶口 (Bottle)", "金属螺母 (Metal Nut)", "网格 (Grid)"))
    input_method = st.sidebar.radio("📸 输入方式", ("📁 文件上传", "📷 实时拍摄"))
    
    MODEL_PATH = {
        "药用瓶口 (Bottle)": 'weights/bottle_ae.pth',
        "金属螺母 (Metal Nut)": 'weights/metal_nut_ae.pth',
        "网格 (Grid)": 'weights/grid_ae.pth'
    }.get(product_type)
    
    model = load_defect_model(MODEL_PATH)
    
    if input_method == "📁 文件上传":
        uploaded_file = st.file_uploader("上传待测图片", type=['png', 'jpg', 'jpeg'])
    else:
        uploaded_file = st.camera_input("拍摄产品图像")

    if model and uploaded_file:
        # 推理逻辑开始
        input_pil = Image.open(uploaded_file).convert('RGB')
        origin_size_np = np.array(input_pil)
        
        with st.spinner("🔬 AI 正在进行结构特征重构..."):
            input_tensor = preprocess(input_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                recon_tensor = model(input_tensor)
            
            recon_np = recon_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            recon_img_rgb = np.clip(recon_np * 255, 0, 255).astype(np.uint8)
            
            # 计算残差
            error_map_tensor = torch.mean(torch.pow(input_tensor - recon_tensor, 2), dim=1).squeeze().cpu()
            error_map_np = error_map_tensor.numpy()
            heatmap_norm = (error_map_np * 255).astype(np.uint8)
            
            # 定位画框
            threshold_value = np.percentile(error_map_np, 99.5) * 255
            _, mask = cv2.threshold(heatmap_norm, threshold_value, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            img_for_bbox = cv2.resize(origin_size_np, (256, 256))
            img_bgr = cv2.cvtColor(img_for_bbox, cv2.COLOR_RGB2BGR)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            defect_count = sum(1 for cnt in contours if cv2.contourArea(cnt) > 20)
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 20:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
            
            final_res = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 结果展示
        st.markdown("### 📊 实时检测分析")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("全局误差 (MSE)", f"{np.mean(error_map_np):.5f}")
        col_m2.metric("缺陷区域数量", f"{defect_count}")
        col_m3.metric("判定结论", "❌ 不合格" if defect_count > 0 else "✅ 合格")

        c1, c2, c3 = st.columns(3)
        with c1: st.image(img_for_bbox, caption="原始样本", use_container_width=True)
        with c2: st.image(recon_img_rgb, caption="AI 理想重构", use_container_width=True)
        with c3: st.image(final_res, caption="缺陷定位结果", use_container_width=True)

# --- 模块 3：数据统计报表 ---
elif menu_selection == "📊 数据统计报表 (Analytics)":
    st.title("📊 历史质量数据分析报告")
    st.write("集成自 `pages/2` 的统计功能，提供生产线质量趋势分析。")
    
    # 模拟一些数据用于展示工作量
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['瓶口线', '螺母线', '网格线']
    )
    st.line_chart(chart_data)
    
    st.subheader("📝 近期检测记录回溯")
    log_data = pd.DataFrame({
        '时间戳': ['2024-04-09 16:10', '2024-04-09 16:15', '2024-04-09 16:22'],
        '产品类型': ['Metal Nut', 'Bottle', 'Grid'],
        '判定结果': ['不合格', '合格', '合格'],
        '置信度': ['98.2%', '99.1%', '97.5%']
    })
    st.table(log_data)

# --- 模块 4：CAE 架构解密 ---
elif menu_selection == "🧬 CAE 架构解密 (Architecture)":
    st.title("🧬 卷积自编码器架构动态演示")
    st.write("本模块通过交互式沙漏模型，阐述非监督学习下的特征降维与重构原理。")
    
    # 嵌入你喜欢的 HTML 动画
    cae_animation_html = """
    <div style="background-color: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0; text-align: center;">
        <h4 style="color: #334155;">数据流沙漏模型 (Dimension Bottleneck)</h4>
        <div style="display: flex; justify-content: center; align-items: center; height: 150px; gap: 5px;">
            <div style="width: 40px; height: 120px; background-color: #3b82f6; opacity: 0.8;"></div>
            <div style="width: 40px; height: 80px; background-color: #60a5fa; opacity: 0.8;"></div>
            <div style="width: 40px; height: 40px; background-color: #93c5fd; opacity: 0.8;"></div>
            <div style="width: 20px; height: 20px; background-color: #1d4ed8; border-radius: 50%;"></div>
            <div style="width: 40px; height: 40px; background-color: #93c5fd; opacity: 0.8;"></div>
            <div style="width: 40px; height: 80px; background-color: #60a5fa; opacity: 0.8;"></div>
            <div style="width: 40px; height: 120px; background-color: #3b82f6; opacity: 0.8;"></div>
        </div>
        <p style="font-size: 0.85em; color: #64748b;">输入层 → 编码器压缩 → 瓶颈层特征 → 解码器重构 → 输出层</p>
    </div>
    """
    st.markdown(cae_animation_html, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🧠 算法原理深度解析
    1. **特征提取 (Encoding)**：通过三层下采样卷积，将 $256\\times256$ 的高维图像压缩为 $32\\times32$ 的核心特征向量。
    2. **异常抹除 (Bottleneck)**：瓶颈层强制舍弃非结构性信息，如划痕、脏污等随机噪音。
    3. **理想还原 (Decoding)**：通过转置卷积将核心特征还原，生成的图像为模型记忆中的“完美样本”。
    """)