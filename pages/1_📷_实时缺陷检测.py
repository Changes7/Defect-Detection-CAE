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

st.set_page_config(page_title="智能检测终端", page_icon="📷", layout="wide", initial_sidebar_state="expanded")

# 1. 数据库写入函数
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

# 2. 核心 AI 视觉处理引擎 (模块化)
def run_cae_pipeline(input_pil, model, device, preprocess, sensitivity):
    origin_size_np = np.array(input_pil) 
    input_tensor = preprocess(input_pil).unsqueeze(0).to(device)
    
    with torch.no_grad(): recon_tensor = model(input_tensor)

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
    
    if max_error_pixel > 0: heatmap_vis = (error_map_np / max_error_pixel * 255).astype(np.uint8)
    else: heatmap_vis = (error_map_np * 255).astype(np.uint8)
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
        "mse": global_mse, "max_err": max_error_pixel, "defect_count": defect_count, "threshold": threshold_value
    }

# 3. 侧边栏配置与路由控制
st.sidebar.markdown("---")
work_mode = st.sidebar.radio("🛠️ 切换工作模式", ["🔍 单图精细检测", "📷 车间实时抓拍", "📊 多图批量评估"])
st.sidebar.markdown("---")

st.sidebar.header("⚙️ 系统设置")
product_type = st.sidebar.selectbox("🔍 选择检测对象模型", ("药用瓶口 (Bottle)", "金属螺母 (Metal Nut)", "网格 (Grid)"))

if product_type == "药用瓶口 (Bottle)": MODEL_PATH = 'weights/bottle_ae.pth'
elif product_type == "金属螺母 (Metal Nut)": MODEL_PATH = 'weights/metal_nut_ae.pth'
else: MODEL_PATH = 'weights/grid_ae.pth' 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sensitivity = st.sidebar.slider("画框敏感度 (Percentile)", min_value=90.0, max_value=99.9, value=99.5, step=0.1)

@st.cache_resource 
def load_defect_model(path): 
    model = ConvAutoEncoder()
    if not os.path.exists(path): return None
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_defect_model(MODEL_PATH)
preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# 4. 动态渲染主界面
st.title("📷 工业产品表面智能检测终端")
st.markdown("---")

if model is None:
    st.error(f"❌ 找不到模型文件 `{MODEL_PATH}`。")
else:
    # 模式 A & B：单图检测 / 摄像头抓拍
    if work_mode in ["🔍 单图精细检测", "📷 车间实时抓拍"]:
        if work_mode == "🔍 单图精细检测":
            input_source = st.file_uploader("上传单张待测图片 (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
        else:
            st.info("💡 提示：点击下方按钮调用摄像头进行实时质检。")
            input_source = st.camera_input("拍照捕获待测样品")

        if input_source is not None:
            input_pil = Image.open(input_source).convert('RGB')
            with st.status("🔬 AI 视觉引擎分析中...", expanded=True) as status:
                st.write("执行特征重构与残差定位...")
                res = run_cae_pipeline(input_pil, model, DEVICE, preprocess, sensitivity)
                status.update(label="✅ 检测完成！", state="complete", expanded=False)

            log_detection_to_db(product_type, res["defect_count"], res["mse"])

            st.markdown("### 📊 量化检测报告")
            m1, m2, m3 = st.columns(3)
            m1.metric(label="全局平均重构误差 (MSE)", value=f"{res['mse']:.5f}")
            m2.metric(label="局部异常峰值", value=f"{res['max_err']:.3f}")
            
            if res["defect_count"] > 0:
                m3.metric(label="综合判定", value="🚨 异常 (不合格)")
                st.error(f"系统警告：锁定 **{res['defect_count']}** 处明显结构异常。")
            else:
                m3.metric(label="综合判定", value="✅ 正常 (合格)")
                st.success("质量放行：未检测到明显表面瑕疵。")
                
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1: st.markdown("**1. 原始样本**"); st.image(res["origin"], use_container_width=True)
            with col2: st.markdown("**2. AI 重构**"); st.image(res["recon"], use_container_width=True)
            with col3: st.markdown("**3. 定位结果**"); st.image(res["result"], use_container_width=True)
                    
            st.markdown("---")
            with st.expander("🔬 极客模式：展开查看 CV 算法底层处理流水线"):
                step1, step2, step3, step4 = st.columns(4)
                with step1: st.markdown("**Step 1: 残差热力图**"); st.image(res["heatmap"], use_container_width=True) 
                with step2: st.markdown("**Step 2: 阈值二值化**"); st.image(res["raw_mask"], use_container_width=True) 
                with step3: st.markdown("**Step 3: 形态学闭运算**"); st.image(res["mask"], use_container_width=True) 
                with step4: st.markdown("**Step 4: 连通域画框**"); st.image(res["result"], use_container_width=True)

    # 模式 C：多图批量评估
    elif work_mode == "📊 多图批量评估":
        st.markdown("### 📥 批量上传测试集")
        uploaded_files = st.file_uploader("请框选多张待测图片", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        
        if uploaded_files:
            st.markdown("### 📊 批量评估结果矩阵")
            h1, h2, h3, h4 = st.columns(4)
            h1.markdown("**1. 原始样本**"); h2.markdown("**2. AI 重构**"); h3.markdown("**3. 残差热力图**"); h4.markdown("**4. 最终定位**")
            st.markdown("---")
            
            for file in uploaded_files:
                input_pil = Image.open(file).convert('RGB')
                res = run_cae_pipeline(input_pil, model, DEVICE, preprocess, sensitivity)
                log_detection_to_db(product_type, res["defect_count"], res["mse"])
                
                c1, c2, c3, c4 = st.columns(4)
                c1.image(res["origin"], use_container_width=True)
                c2.image(res["recon"], use_container_width=True)
                c3.image(res["heatmap"], use_container_width=True)
                c4.image(res["result"], use_container_width=True)
                st.markdown("<hr style='margin: 10px 0; border: 0.5px solid #e0e0e0'>", unsafe_allow_html=True)