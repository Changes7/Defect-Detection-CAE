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

st.set_page_config(page_title="检测终端", page_icon="📷", layout="wide")

def log_to_db(prod, defect, mse):
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/industrial_inspection.db")
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS inspection_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, product_type TEXT, result TEXT, defect_count INTEGER, mse REAL)')
    res = "NG" if defect > 0 else "OK"
    cursor.execute('INSERT INTO inspection_logs (timestamp, product_type, result, defect_count, mse) VALUES (?, ?, ?, ?, ?)', (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), prod, res, defect, mse))
    conn.commit()
    conn.close()

def run_ai_engine(img, model, device, prep, sens):
    # AI 推理过程
    input_t = prep(img).unsqueeze(0).to(device)
    with torch.no_grad(): 
        recon_t = model(input_t)
    
    # 后处理
    recon_np = np.clip(recon_t.squeeze().permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    err_map = torch.mean(torch.pow(input_t - recon_t, 2), dim=1).squeeze().cpu().numpy()
    mse = float(np.mean(err_map))
    
    thresh = np.percentile(err_map, sens)
    mask = cv2.morphologyEx(np.where(err_map > thresh, 255, 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    
    # 画框与水印
    res_bgr = cv2.cvtColor(cv2.resize(np.array(img), (256, 256)), cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    defects = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 20:
            defects += 1
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(res_bgr, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.putText(res_bgr, f"#{defects}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
            
    res_text = "NG (Abnormal)" if defects > 0 else "OK (Normal)"
    res_color = (0,0,255) if defects > 0 else (0,255,0)
    cv2.putText(res_bgr, f"RESULT: {res_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, res_color, 2)
    
    heatmap = cv2.applyColorMap((err_map / (np.max(err_map)+1e-7) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    return {
        "orig": cv2.resize(np.array(img), (256, 256)),
        "recon": recon_np,
        "heat": cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB),
        "res": cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB),
        "mse": mse,
        "defects": defects
    }

st.sidebar.markdown("### 📷 检测配置")
mode = st.sidebar.radio("工作模式", ["🔍 单图检测", "📷 拍照检测", "📊 多图批量评估"])
prod = st.sidebar.selectbox("产品模型", ("Bottle", "Metal Nut", "Grid"))
sens = st.sidebar.slider("画框敏感度 (Percentile)", 90.0, 99.9, 99.5)
st.sidebar.markdown("---")
st.sidebar.info("[📁 GitHub 仓库](https://github.com/Changes7/Defect-Detection-CAE)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource 
def get_model(p):
    path = f"weights/{p.lower().replace(' ', '_')}_ae.pth"
    if not os.path.exists(path): return None
    m = ConvAutoEncoder()
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    return m.to(DEVICE).eval()

model = get_model(prod)
prep = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

st.title("📷 智能质检终端")
if model is None: st.error("❌ 权重文件丢失，请检查 weights 文件夹")
else:
    files = st.file_uploader("上传图片", accept_multiple_files=(mode=="📊 多图批量评估"))
    if mode == "📷 拍照检测": files = st.camera_input("拍照")
    
    if files:
        f_list = files if isinstance(files, list) else [files]
        if mode == "📊 多图批量评估": st.subheader("📊 批量评估结果矩阵")

        for f in f_list:
            res = run_ai_engine(Image.open(f).convert('RGB'), model, DEVICE, prep, sens)
            log_to_db(prod, res["defects"], res["mse"])
            
            if mode == "📊 多图批量评估":
                # 批量评估：四列并排，每张都有 caption
                c1, c2, c3, c4 = st.columns(4)
                c1.image(res["orig"], caption="[原始样本]", use_container_width=True)
                c2.image(res["recon"], caption="[AI 重构]", use_container_width=True)
                c3.image(res["heat"], caption="[热力残差]", use_container_width=True)
                c4.image(res["res"], caption="[判定结果]", use_container_width=True)
                st.markdown("<hr style='margin:15px 0; border:0.5px solid #eee'>", unsafe_allow_html=True)
            else:
                # 单图/拍照：三列并排，每张都有 caption
                col1, col2, col3 = st.columns(3)
                col1.image(res["orig"], caption="原始样本", use_container_width=True)
                col2.image(res["recon"], caption="AI 重构", use_container_width=True)
                col3.image(res["res"], caption="判定结果", use_container_width=True)
                with st.expander("🔬 查看热力分析"):
                    st.image(res["heat"], caption="残差热力分布图", width=400)
                st.markdown("---")