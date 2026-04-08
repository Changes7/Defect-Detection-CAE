import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from model import ConvAutoEncoder # 引入你的模型定义

# ==========================================
# 0. 全局页面配置 (必须放在最前面)
# ==========================================
st.set_page_config(
    page_title="工业缺陷检测系统", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. 动态日夜主题引擎 (UI 美化与状态管理)
# ==========================================
# 在侧边栏顶部添加一个精美的开关
theme_toggle = st.sidebar.toggle("🌞 / 🌙 切换深夜工业模式", value=True)

# 根据开关状态，动态分配色彩变量
if theme_toggle:
    # 🌙 深夜工业模式 (Dark Mode)
    bg_color = "#0F172A"
    sidebar_bg = "#1E293B"
    text_color = "#F8FAFC"
    card_bg = "#1E293B"
    border_color = "#334155"
else:
    # 🌞 白天明亮模式 (Light Mode)
    bg_color = "#F8FAFC"
    sidebar_bg = "#F1F5F9"
    text_color = "#0F172A"
    card_bg = "#FFFFFF"
    border_color = "#E2E8F0"

# 动态组装 CSS 样式表 (【已修复】保留 header，防止侧边栏无法打开)
dynamic_css = f"""
<style>
/* 隐藏默认水印，但保留 header 以便打开侧边栏 */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* 强制覆盖主页面和侧边栏背景 */
[data-testid="stAppViewContainer"] {{
    background-color: {bg_color} !important;
}}
[data-testid="stSidebar"] {{
    background-color: {sidebar_bg} !important;
}}

/* 覆盖各级标题和正文颜色 */
h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{
    color: {text_color} !important;
}}

/* 仪表盘卡片高级定制 (带悬浮浮动效果) */
[data-testid="stMetric"] {{
    background-color: {card_bg} !important;
    border-radius: 10px;
    padding: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid {border_color};
    transition: all 0.3s ease;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
}}

/* 图片圆角处理 */
img {{
    border-radius: 8px;
}}
</style>
"""
# 注入动态 CSS
st.markdown(dynamic_css, unsafe_allow_html=True)


# ==========================================
# 2. 侧边栏：操作面板与参数设置
# ==========================================
st.sidebar.markdown("---")
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
    
    # 读取原始图片
    input_pil = Image.open(uploaded_file).convert('RGB')
    origin_size_np = np.array(input_pil) 

    # === 高级状态栏：沉浸式检测过程 ===
    with st.status("🔬 初始化 AI 视觉引擎...", expanded=True) as status:
        st.write("📥 正在预处理图像张量...")
        input_tensor = preprocess(input_pil).unsqueeze(0).to(DEVICE)
        
        st.write("🧠 自编码器特征重构中...")
        with torch.no_grad():
            recon_tensor = model(input_tensor)

        # 获取 Numpy 格式用于 OpenCV 处理
        img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        recon_np = recon_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

        st.write("🎯 提取残差并执行阈值分割...")
        # 计算 MSE 残差图
        error_map_tensor = torch.mean(torch.pow(input_tensor - recon_tensor, 2), dim=1).squeeze().cpu()
        error_map_np = error_map_tensor.numpy()
        heatmap_norm = (error_map_np * 255).astype(np.uint8)

        # 提取量化数据
        global_mse = float(np.mean(error_map_np))
        max_error_pixel = float(np.max(error_map_np))

        st.write("📐 生成热力图与自动定位画框...")
        
        # 自动化缺陷画框
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
                # 绘制高亮纯红框，加粗至 3
                cv2.rectangle(img_for_bbox_bgr, (x, y), (x+w, y+h), (0, 0, 255), 3)
                defect_count += 1
        
        img_with_boxes_rgb = cv2.cvtColor(img_for_bbox_bgr, cv2.COLOR_BGR2RGB)
        
        # 将灰度的残差图渲染成 Jet 伪彩色热力图
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        # 进度条收尾
        status.update(label="✅ 检测完成！", state="complete", expanded=False)

    # --- 阶段 E：网页可视化展示 ---
    st.markdown("### 📊 量化检测报告")
    
    # 工业级数据仪表盘
    m1, m2, m3 = st.columns(3)
    m1.metric(label="全局平均重构误差 (MSE)", value=f"{global_mse:.5f}")
    m2.metric(label="局部异常峰值", value=f"{max_error_pixel:.3f}")
    
    if defect_count > 0:
        m3.metric(label="系统综合判定", value="🚨 异常 (不合格)")
        st.error(f"系统警告：共锁定 **{defect_count}** 处明显结构异常。")
    else:
        m3.metric(label="系统综合判定", value="✅ 正常 (合格)")
        st.success("质量放行：未检测到明显表面瑕疵。")
        st.toast('检测放行：产品表面完好', icon='✅') # 右下角轻提示
        
    st.markdown("<br>", unsafe_allow_html=True)

    # 图像对比矩阵 (使用 width="stretch" 防止终端报警)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. 原始样本 (Input)**")
        st.image(img_for_bbox, width="stretch")
        
    with col2:
        st.markdown("**2. 残差热力分析 (Heatmap)**")
        st.image(heatmap_color_rgb, width="stretch")
        
    with col3:
        st.markdown("**3. 定位结果 (Defect Box)**")
        st.image(img_with_boxes_rgb, width="stretch")

else:
    # ==========================================
    # 5. 初始欢迎页与训练收敛图
    # ==========================================
    st.info("👈 系统就绪。请在左侧控制台选择模型类型，并上传产品图像进行分析。")
    st.markdown("---")
    st.markdown("### 📈 算法模型收敛状态 (Training Loss)")
    
    if os.path.exists('results/loss_curve_result.png'):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image('results/loss_curve_result.png', caption="CAE 模型训练误差收敛曲线", width="stretch")
    else:
        st.warning("暂无模型训练数据图表 (results/loss_curve_result.png)")