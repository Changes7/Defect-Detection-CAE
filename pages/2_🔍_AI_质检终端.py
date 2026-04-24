import streamlit as st
import numpy as np
import cv2
import os
import io
import requests
from PIL import Image
from core_utils import log_to_db  
from style_utils import apply_glass_theme

apply_glass_theme()

# ==========================================
# 0. 企业级安全沙箱 (RBAC 登录拦截)
# ==========================================
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.error("🛑 拒绝访问：您尚未登录系统，请先前往主页进行身份认证。")
    st.stop()

# 隐藏顶部空白，增加可视区域
st.markdown("<style> .block-container {padding-top: 2rem;} </style>", unsafe_allow_html=True)
st.title("🔍 AI 智能质检工位 (云端推理版)")
PROD_MAP = {"药用瓶口 (Bottle)": "bottle", "金属螺母 (Nut)": "metal_nut", "网格 (Grid)": "grid", "螺丝 (Screw)": "screw", "药片 (Pill)": "pill"}

os.makedirs("feedback_negatives", exist_ok=True)

# ==========================================
# 1. 左侧操作面板 (彻底释放主视口空间)
# ==========================================
with st.sidebar:
    st.markdown("### 🎛️ 操作面板")
    selected = st.selectbox("🎯 检测对象", list(PROD_MAP.keys()))
    
    st.markdown("---")
    mode = st.radio("📂 数据来源", ["📁 本地上传", "📸 实时拍摄", "🖼️ 示例图库 (快速体验)"])
    
    st.markdown("---")
    st.markdown("#### ⚙️ 算法高级设置")
    sens = st.slider("📏 画框敏感度", 90.0, 99.9, 99.5)
    alpha = st.slider("👁️ 热力图透视层叠度", 0.0, 1.0, 0.6, step=0.05)
    api_host = st.text_input("🔗 后端引擎 API", value="http://127.0.0.1:8000")
    
    st.markdown("---")
    st.info("💡 **检测原理**：\n1. 自编码器重建正常特征。\n2. 差分计算提取异常区域。\n3. OpenCV 定位标记缺陷。")

# ==========================================
# 2. 主视觉区：数据源获取
# ==========================================
uploaded_files = []

if mode == "🖼️ 示例图库 (快速体验)":
    st.info("💡 点击下方预置样本，一键体验微服务推理。")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    if ex_col1.button("🔩 测试样本 1 (良品)", use_container_width=True):
        uploaded_files = ["examples/sample_ok.jpg"] if os.path.exists("examples/sample_ok.jpg") else []
    if ex_col2.button("🔩 测试样本 2 (细微划痕)", use_container_width=True):
        uploaded_files = ["examples/sample_ng1.jpg"] if os.path.exists("examples/sample_ng1.jpg") else []
    if ex_col3.button("🔩 测试样本 3 (严重变形)", use_container_width=True):
        uploaded_files = ["examples/sample_ng2.jpg"] if os.path.exists("examples/sample_ng2.jpg") else []
        
    if not uploaded_files:
        st.warning("⚠️ 未找到示例图片，请在项目根目录创建 `examples` 文件夹并放入图片。")

elif mode == "📁 本地上传":
    files = st.file_uploader("第一步：上传待测图片", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    if files: uploaded_files = files
else:
    cam_file = st.camera_input("📸 工业视觉实时抓拍通道")
    if cam_file: uploaded_files = [cam_file]
    
# ==========================================
# 3. 核心推理与 3 列结果展示
# ==========================================
if uploaded_files:
    st.success(f"✅ 任务建立！共收到 {len(uploaded_files)} 个检测任务，正在呼叫云端引擎...")
    st.divider()
    
    for idx, file in enumerate(uploaded_files):
        file_name = file.name if hasattr(file, 'name') else str(file).split('/')[-1]
        st.markdown(f"#### 📄 样本编号: `{file_name}`")
        
        # 预处理图像
        img = Image.open(file).convert('RGB')
        orig_res = cv2.resize(np.array(img), (256, 256))
        orig_rgb = orig_res.copy() # 用于图1展示
        res_bgr = cv2.cvtColor(orig_res, cv2.COLOR_RGB2BGR) # 用于图3画框

        with st.spinner("🚀 正在将图像通过 RESTful API 发送至推理引擎..."):
            try:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                img_bytes = img_byte_arr.getvalue()

                api_url = f"{api_host}/api/v1/predict?selected_product={selected}&sens={sens}"
                response = requests.post(api_url, files={"file": ("image.jpg", img_bytes, "image/jpeg")})
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        mse = result["mse"]
                        max_err = result["max_err"]
                        defects = result["defects_count"]
                        bboxes = result["bboxes"]
                        err_map = np.array(result["err_map"]) 
                        
                        log_to_db(selected, defects, mse)  
                        
                        # 画框逻辑
                        for box in bboxes:
                            x, y, w, h = box
                            cv2.rectangle(res_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        
                        # 热力图合成逻辑
                        heatmap_bgr = cv2.applyColorMap((err_map / (np.max(err_map)+1e-7) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                        blended_img = cv2.addWeighted(orig_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
                        
                        # ------------------------------------------
                        # 重点重构：图二风格的 3 列排版
                        # ------------------------------------------
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**1. 原始待测图**")
                            st.image(orig_rgb, use_container_width=True)
                            
                        with col2:
                            st.markdown("**2. 深度异常分析图**")
                            st.image(blended_img, use_container_width=True)
                            
                        with col3:
                            st.markdown("**3. 最终检测定位结果**")
                            st.image(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        # ------------------------------------------
                        # 判定结果与反馈条放置在图片下方
                        # ------------------------------------------
                        if defects == 0:
                            severity_score = min(mse * 15000, 30) 
                            st.success(f"🟢 **【判定结果：合格】未检测到明显缺陷，准予放行。** (异常指数: {severity_score:.1f}/100)")
                        else:
                            severity_score = min(60 + (defects * 8) + (max_err * 500), 100)
                            st.error(f"🔴 **【判定结果：不合格】检测到 {defects} 处缺陷，请剔除！** (异常指数: {severity_score:.1f}/100)")
                        
                        # 误判反馈按钮单独一行对齐
                        col_space, col_btn = st.columns([4, 1])
                        with col_btn:
                            if st.button("👎 误判反馈 (入库)", key=f"fb_btn_{idx}", use_container_width=True):
                                save_path = f"feedback_negatives/fb_{selected}_{idx}.jpg"
                                img.save(save_path)
                                st.toast("✅ 已记录至主动学习队列。", icon="📈")
                        
                    else:
                        st.error(f"⚠️ 引擎返回错误: {result.get('message')}")
                else:
                    st.error(f"⛔ 网络请求失败，状态码: {response.status_code}。请检查 API 参数！")
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 **无法连接到推理引擎！**")
                st.warning("请确保你已经打开了另一个终端窗口，并运行了后端的 uvicorn 或 PM2 进程。")
            except Exception as e:
                st.error(f"发生未知错误: {str(e)}")
                
        st.markdown("<hr style='border: 1px dashed #cbd5e1; margin-top: 2rem;'>", unsafe_allow_html=True)