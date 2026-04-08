import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from model import ConvAutoEncoder # 引入你的模型定义

# ==========================================
# 1. 全局配置与 UI 美化 (必须放在最前)
# ==========================================
st.set_page_config(
    page_title="工业缺陷检测系统", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 隐藏 Streamlit 默认水印，提升专业度
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# 2. 侧边栏：操作面板与参数设置
# ==========================================
st.sidebar.header("⚙️ 系统设置")
product_type = st.sidebar.selectbox(
    "🔍 选择检测对象模型",
    ("药用瓶口 (Bottle)", "金属螺母 (Metal Nut)")
)

# 动态绑定权重路径
MODEL_PATH = 'weights/bottle_ae.pth' if product_type == "药用瓶口 (Bottle)" else 'weights/metal_nut_ae.pth'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.sidebar.markdown("---")
st.sidebar.header("📥 数据输入")
uploaded_file = st.sidebar.file_uploader("上传待测图片 (PNG/JPG)", type=['png', 'jpg', 'jpeg'])

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 检测原理说明
1. **输入样本**：传入全新图像。
2. **AI 重构**：自编码器抹除异常特征。
3. **残差计算**：原图与重构图作差。
4. **自动定位**：OpenCV 锁定瑕疵区域。
""")

# ==========================================
# 3. 核心加载逻辑 (带缓存加速)
# ==========================================
@st.cache_resource 
def load_defect_model(path): 
    # 传入 path 参数，这样切换下拉菜单时，Streamlit 会自动加载新模型
    model = ConvAutoEncoder()
    if not os.path.exists(path):
        return None
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_defect_model(MODEL_PATH)

# 核心预处理流程
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ==========================================
# 4. 主界面：仪表盘与图像展示
# ==========================================
st.title("🤖 工业产品表面缺陷自动化检测系统 (CAE-BN Prototype)")
st.markdown("---")

if model is None:
    st.error(f"❌ 严重错误：找不到模型文件 `{MODEL_PATH}`。请确保模型已训练并放在正确的目录下。")
elif uploaded_file is not None:
    # --- 阶段 A：图像预处理 ---
    input_pil = Image.open(uploaded_file).convert('RGB')
    origin_size_np = np.array(input_pil) 

    with st.spinner('⚡ AI 正在进行张量运算与残差分析...'):
        input_tensor = preprocess(input_pil).unsqueeze(0).to(DEVICE)
        
        # --- 阶段 B：模型推理 ---
        with torch.no_grad():
            recon_tensor = model(input_tensor)

        # 获取 Numpy 格式用于 OpenCV 处理
        img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        recon_np = recon_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

        # 计算 MSE 残差热力图
        error_map_tensor = torch.mean(torch.pow(input_tensor - recon_tensor, 2), dim=1).squeeze().cpu()
        error_map_np = error_map_tensor.numpy()
        heatmap_norm = (error_map_np * 255).astype(np.uint8)

        # --- 阶段 C：提取量化数据 (新增：用于仪表盘展示) ---
        global_mse = float(np.mean(error_map_np))
        max_error_pixel = float(np.max(error_map_np))

        # --- 阶段 D：自动化缺陷画框 ---
        threshold_value = np.percentile(error_map_np, 99.5) * 255
        _, mask = cv2.threshold(heatmap_norm, threshold_value, 255, cv2.THRESH_BINARY)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        img_for_bbox = cv2.resize(origin_size_np, (256, 256))
        img_for_bbox_bgr = cv2.cvtColor(img_for_bbox, cv2.COLOR_RGB2BGR)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defect_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 20: 
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img_for_bbox_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2)
                defect_count += 1
        
        img_with_boxes_rgb = cv2.cvtColor(img_for_bbox_bgr, cv2.COLOR_BGR2RGB)

    # --- 阶段 E：网页可视化展示 ---
    st.markdown("### 📊 量化检测报告")
    
    # 1. 工业级数据仪表盘
    m1, m2, m3 = st.columns(3)
    m1.metric(label="全局平均重构误差 (MSE)", value=f"{global_mse:.5f}")
    m2.metric(label="局部异常峰值", value=f"{max_error_pixel:.3f}")
    
    if defect_count > 0:
        m3.metric(label="系统综合判定", value="🚨 异常 (不合格)")
        st.error(f"系统警告：共锁定 **{defect_count}** 处明显结构异常。")
    else:
        m3.metric(label="系统综合判定", value="✅ 正常 (合格)")
        st.success("质量放行：未检测到明显表面瑕疵。")
        
    st.markdown("<br>", unsafe_allow_html=True)

    # 2. 图像对比矩阵
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. 原始样本 (Input)**")
        st.image(img_for_bbox, use_container_width=True) # 使用最新的 use_container_width
    with col2:
        st.markdown("**2. AI 重构 (Reconstruction)**")
        recon_show = recon_np.clip(0, 1)
        st.image(recon_show, use_container_width=True)
    with col3:
        st.markdown("**3. 定位结果 (Defect Box)**")
        st.image(img_with_boxes_rgb, use_container_width=True)

else:
    # ==========================================
    # 5. 初始欢迎页与训练收敛图
    # ==========================================
    st.info("👈 系统就绪。请在左侧控制台选择模型类型，并上传产品图像进行分析。")
    st.markdown("---")
    st.markdown("### 📈 算法模型收敛状态 (Training Loss)")
    
    if os.path.exists('results/loss_curve_result.png'):
        # 居中显示收敛图
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image('results/loss_curve_result.png', caption="CAE 模型训练误差收敛曲线", use_container_width=True)
    else:
        st.warning("暂无模型训练数据图表 (results/loss_curve_result.png)")