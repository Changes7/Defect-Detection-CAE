import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
from model import ConvAutoEncoder # 引入你的模型定义

# --- 1. 全局配置与模型加载 ---
st.set_page_config(page_title="工业缺陷检测系统 Demo", layout="wide")

MODEL_PATH = 'weights/bottle_ae.pth' # 确保模型在这个路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 缓存加载模型，防止每次刷新页面都重复加载，提升性能
@st.cache_resource
def load_defect_model():
    model = ConvAutoEncoder()
    if not os.path.exists(MODEL_PATH):
        st.error(f"错误：找不到模型文件 {MODEL_PATH}，请确保已训练并整理了项目结构。")
        return None
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_defect_model()

# 核心预处理流程 (与训练时完全一致)
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# --- 2. 网页 UI 布局 ---
st.title("🤖 工业产品表面缺陷自动化检测系统 (CAE-BN Prototype)")
st.markdown("---")

st.sidebar.header("操作面板")
uploaded_file = st.sidebar.file_uploader("第一步：上传瓶口待测图片", type=['png', 'jpg', 'jpeg'])
st.sidebar.markdown("""
### 检测原理说明
1. **上传图片**：输入一个全新的待测样本。
2. **AI 重构**：自编码器尝试将其还原为记忆中的“正常样本”。
3. **残差计算**：比较原图与重构图的差异。
4. **自动画框**：通过 OpenCV 定位差异过大的瑕疵区域。
""")

if uploaded_file is not None:
    # 3. 如果上传了文件，开始执行检测流程
    # 3.1 读取 PIL 图片
    input_pil = Image.open(uploaded_file).convert('RGB')
    origin_size_np = np.array(input_pil) # 保留原始尺寸供后期画框

    with st.spinner('AI 正在紧张检测中，请稍候...'):
        # 3.2 预处理并运行推理
        input_tensor = preprocess(input_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            recon_tensor = model(input_tensor)

        # 3.3 计算残差热力图 (这里复用 test.py 的核心 OpenCV 逻辑)
        # 获取 Tensor 并转回 Numpy
        img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
        recon_np = recon_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

        # 计算 MSE 残差图
        error_map_tensor = torch.mean(torch.pow(input_tensor - recon_tensor, 2), dim=1).squeeze().cpu()
        error_map_np = error_map_tensor.numpy()
        heatmap_norm = (error_map_np * 255).astype(np.uint8)

        # --- 3.4 自动化缺陷检测与画框 (核心改进部分) ---
        # 复用 test.py 验证有效的动态阈值
        threshold_value = np.percentile(error_map_np, 99.5) * 255
        _, mask = cv2.threshold(heatmap_norm, threshold_value, 255, cv2.THRESH_BINARY)
        
        # 形态学闭运算，让瑕疵连成片
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 准备原图副本用于画框 (调整尺寸为 256x256 匹配模型)
        img_for_bbox = cv2.resize(origin_size_np, (256, 256))
        img_with_boxes_rgb = img_for_bbox.copy()
        img_for_bbox_bgr = cv2.cvtColor(img_for_bbox, cv2.COLOR_RGB2BGR)

        # 寻找轮廓并画红框
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defect_count = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > 20: 
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img_for_bbox_bgr, (x, y), (x+w, y+h), (255, 0, 0), 2) # BGR
                defect_count += 1
        
        # 转换回 RGB 用于 Streamlit 显示
        img_with_boxes_rgb = cv2.cvtColor(img_for_bbox_bgr, cv2.COLOR_BGR2RGB)

    # --- 4. 网页结果可视化展示 ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("1. 原始待测图")
        st.image(img_for_bbox, use_column_width=True)

    with col2:
        st.subheader("2. AI 重构理想图")
        # 转换回可显示的 Numpy 格式
        recon_show = recon_np.clip(0, 1)
        st.image(recon_show, use_column_width=True)

    with col3:
        st.subheader("3. 最终检测定位结果")
        st.image(img_with_boxes_rgb, use_column_width=True)
        # 添加显著的判定状态
        if defect_count > 0:
            st.error(f"🚫 检测到缺陷 ( Found {defect_count}瑕疵 )，判定：【不合格】")
        else:
            st.success("✅ 未检测到明显瑕疵，判定：【合格】")

else:
    # 5. 初始状态：显示一个欢迎页
    st.warning("👈 请在左侧面板上传瓶口图片开始检测。")
    # 为了演示效果，建议你也贴上你的 Loss 曲线图
    st.markdown("---")
    st.subheader("当前项目训练收敛情况 (Loss Curve)")
    if os.path.exists('results/loss_curve_result.png'):
        st.image('results/loss_curve_result.png', width=700)
    else:
        st.text("Loss 曲线图未生成。")