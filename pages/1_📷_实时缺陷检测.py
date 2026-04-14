import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import datetime
import sqlite3
from model import ConvAutoEncoder 

# ==========================================
# 0. 全局页面配置
# ==========================================
st.set_page_config(page_title="智能检测终端", page_icon="📷", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# 1. 数据库写入函数
# ==========================================
def log_detection_to_db(product_type, defect_count, mse):
    db_dir = "data"
    db_path = os.path.join(db_dir, "industrial_inspection.db")
    if not os.path.exists(db_dir): os.makedirs(db_dir)
        
    result_str = "NG" if defect_count > 0 else "OK"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS inspection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            product_type TEXT,
            result TEXT,
            defect_count INTEGER,
            mse REAL
        )
    ''')
    cursor.execute('''
        INSERT INTO inspection_logs (timestamp, product_type, result, defect_count, mse)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, product_type, result_str, defect_count, mse))
    conn.commit()
    conn.close()

# ==========================================
# 2. 核心 AI 视觉处理引擎
# ==========================================
def run_cae_pipeline(input_pil, model, device, preprocess, sensitivity):
    origin_size_np = np.array(input_pil) 
    input_tensor = preprocess(input_pil).unsqueeze(0).to(device)
    
    with torch.no_grad(): 
        recon_tensor = model(input_tensor)

    recon_np = recon_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    recon_img_rgb = np.clip(recon_np * 255, 0, 255).astype(np.uint8)

    error_map_tensor = torch.mean(torch.pow(input_tensor - recon_tensor, 2), dim=1).squeeze().cpu()
    error_map_np = error_map_tensor.numpy()
    
    global_mse = float(np.mean(error_map_np))
    max_error_pixel = float(np.max(error_map_np))

    threshold_value = np.percentile(error_map_np, sensitivity)
    mask = np.where(error_map_np > threshold_value, 255, 0).astype(np.uint8)
    raw_mask = mask.copy()
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_for_bbox = cv2.resize(origin_size_np, (256, 256))
    img_for_bbox_bgr = cv2.cvtColor(img_for_bbox, cv2.COLOR_RGB2BGR)
    
    heatmap_vis = (error_map_np / (max_error_pixel + 1e-8) * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defect_count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 20: 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_for_bbox_bgr, (x, y), (x+w, y+h), (0, 0, 255), 3)
            defect_count += 1
    
    img_with_boxes_rgb = cv2.cvtColor(img_for_bbox_bgr, cv2.COLOR_BGR2RGB)
    return {
        "origin": img_for_bbox, "recon": recon_img_rgb, "heatmap": heatmap_rgb, 
        "result": img_with_boxes_rgb, "raw_mask": raw_mask, "mask": mask,
        "mse": global_mse, "max_err": max_error_pixel, "defect_count": defect_count
    }

# ==========================================
# 3. 侧边栏配置 (同步 GitHub 入口)
# ==========================================
st.sidebar.markdown("### 🛠️ 工作台设置")
work_mode = st.sidebar.radio("切换工作模式", ["🔍 单图精细检测", "📷 车间实时抓拍", "📊 多图批量评估"])
st.sidebar.markdown("---")

st.sidebar.header("⚙️ 算法配置")
product_type = st.sidebar.selectbox("选择检测模型", ("药用瓶口 (Bottle)", "金属螺母 (Metal Nut)", "网格 (Grid)"))

if product_type == "药用瓶口 (Bottle)": MODEL_PATH = 'weights/bottle_ae.pth'
elif product_type == "金属螺母 (Metal Nut)": MODEL_PATH = 'weights/metal_nut_ae.pth'
else: MODEL_PATH = 'weights/grid_ae.pth' 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sensitivity = st.sidebar.slider("画框敏感度 (Percentile)", 90.0, 99.9, 99.5, 0.1)

# 源码入口链接
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔗 源码与资源")
st.sidebar.info("[📁 GitHub: Defect-Detection-CAE](https://github.com/Changes7/Defect-Detection-CAE)")

@st.cache_resource 
def load_defect_model(path): 
    if not os.path.exists(path): return None
    model = ConvAutoEncoder()
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

model = load_defect_model(MODEL_PATH)
preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# ==========================================
# 4. 主界面渲染
# ==========================================
st.title("📷 工业产品表面智能检测终端")
st.markdown("---")

if model is None:
    st.error(f"❌ 权重文件缺失：无法在路径 `{MODEL_PATH}` 找到模型。请确认 GitHub 仓库 weights 目录正确。")
else:
    if work_mode in ["🔍 单图精细检测", "📷 车间实时抓拍"]:
        if work_mode == "🔍 单图精细检测":
            input_source = st.file_uploader("上传待测图片", type=['png', 'jpg', 'jpeg'])
        else:
            input_source = st.camera_input("调用车间摄像头抓拍")

        if input_source:
            input_pil = Image.open(input_source).convert('RGB')
            with st.status("🔬 AI 视觉引擎分析中...") as status:
                res = run_cae_pipeline(input_pil, model, DEVICE, preprocess, sensitivity)
                status.update(label="✅ 检测完成", state="complete")

            log_detection_to_db(product_type, res["defect_count"], res["mse"])

            st.markdown("### 📊 量化报告")
            m1, m2, m3 = st.columns(3)
            m1.metric("重构误差 (MSE)", f"{res['mse']:.5f}")
            m2.metric("异常峰值", f"{res['max_err']:.3f}")
            m3.metric("判定结果", "🚨 异常" if res["defect_count"] > 0 else "✅ 合格")
                
            c1, c2, c3 = st.columns(3)
            c1.image(res["origin"], caption="原始样本", use_container_width=True)
            c2.image(res["recon"], caption="AI 重构", use_container_width=True)
            c3.image(res["result"], caption="定位结果", use_container_width=True)
            
            with st.expander("🔬 极客模式：算法处理细节"):
                s1, s2, s3 = st.columns(3)
                s1.image(res["heatmap"], caption="残差热力图")
                s2.image(res["raw_mask"], caption="阈值二值化")
                s3.image(res["mask"], caption="闭运算增强")

    elif work_mode == "📊 多图批量评估":
        uploaded_files = st.file_uploader("框选多张图片进行并行评估", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                input_pil = Image.open(file).convert('RGB')
                res = run_cae_pipeline(input_pil, model, DEVICE, preprocess, sensitivity)
                log_detection_to_db(product_type, res["defect_count"], res["mse"])
                
                c1, c2, c3, c4 = st.columns(4)
                c1.image(res["origin"], use_container_width=True)
                c2.image(res["recon"], use_container_width=True)
                c3.image(res["heatmap"], use_container_width=True)
                c4.image(res["result"], use_container_width=True)
                st.markdown("---")