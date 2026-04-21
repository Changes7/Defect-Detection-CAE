import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from core_utils import load_model, log_to_db, DEVICE

st.title("🔍 AI 智能质检工位")
PROD_MAP = {"药用瓶口 (Bottle)": "bottle", "金属螺母 (Nut)": "metal_nut", "网格 (Grid)": "grid", "螺丝 (Screw)": "screw", "药片 (Pill)": "pill"}
preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

col_set, col_run = st.columns([1, 3])
with col_set:
    st.write("🔧 检测设置")
    selected = st.selectbox("检测对象", list(PROD_MAP.keys()))
    mode = st.radio("来源", ["📁 文件上传", "📷 实时拍摄"])
    sens = st.slider("画框敏感度", 90.0, 99.9, 99.5)

with col_run:
    if mode == "📁 文件上传":
        # 核心改动 1：开启 accept_multiple_files=True
        uploaded_files = st.file_uploader("上传图片 (支持按住 Ctrl 多选或拖拽多张)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    else:
        # 如果是拍照模式，Streamlit 默认只支持单张，我们把它包成列表，方便统一使用循环处理
        cam_file = st.camera_input("拍照")
        uploaded_files = [cam_file] if cam_file else []
        
    if uploaded_files:
        model_path = f"weights/{PROD_MAP[selected]}_ae.pth"
        model = load_model(model_path)
        
        if model is None:
            st.error(f"❌ 找不到模型权重文件: {model_path}")
        else:
            st.success(f"✅ 模型就绪！共收到 {len(uploaded_files)} 个检测任务，正在批量处理...")
            
            # 核心改动 2：加入 for 循环遍历所有上传的图片
            for idx, file in enumerate(uploaded_files):
                # 打印当前正在处理的文件名
                file_name = file.name if hasattr(file, 'name') else "来自摄像头的抓拍"
                st.markdown(f"#### 📄 样本 {idx + 1}: `{file_name}`")
                
                img = Image.open(file).convert('RGB')
                with st.spinner("🔬 AI 正在分析..."):
                    t = preprocess(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad(): output = model(t)
                    
                    recon_np = np.clip(output.squeeze().permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                    err_map = torch.mean(torch.pow(t - output, 2), dim=1).squeeze().cpu().numpy()
                    mse = float(np.mean(err_map))
                    
                    mask = cv2.morphologyEx(np.where(err_map > np.percentile(err_map, sens), 255, 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
                    orig_res = cv2.resize(np.array(img), (256, 256))
                    res_bgr = cv2.cvtColor(orig_res, cv2.COLOR_RGB2BGR)
                    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    defects = sum(1 for c in contours if cv2.contourArea(c) > 20)
                    for c in contours:
                        if cv2.contourArea(c) > 20:
                            x, y, w, h = cv2.boundingRect(c)
                            cv2.rectangle(res_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    
                    # 将每张图的结果独立写入数据库
                    log_to_db(selected, defects, mse)  
                
                # 为每张图片渲染独立的检测结果
                if defects > 0:
                    st.error(f"🚨 判定：【❌ 不合格】 | 锁定缺陷：**{defects}** 处 | MSE：{mse:.4f}")
                else:
                    st.success(f"✅ 判定：【✅ 合格】 | 未见明显瑕疵 | MSE：{mse:.4f}")
                    
                c1, c2, c3 = st.columns(3)
                c1.image(orig_res, caption="原始样本", use_container_width=True)
                c2.image(recon_np, caption="AI 重构", use_container_width=True)
                c3.image(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB), caption="检测结果", use_container_width=True)
                
                # 加一条分割线，区分多张图片
                st.markdown("<hr style='border: 1px dashed #cbd5e1;'>", unsafe_allow_html=True)