import streamlit as st
import numpy as np
import cv2
import os
import io
import requests
from PIL import Image
from core_utils import log_to_db  # 前端仅保留轻量级的数据库履历写入功能
from style_utils import apply_glass_theme
apply_glass_theme()

# ==========================================
# 0. 企业级安全沙箱 (RBAC 登录拦截)
# ==========================================
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.error("🛑 拒绝访问：您尚未登录系统，请先前往主页进行身份认证。")
    st.stop()

st.title("🔍 AI 智能质检工位 (云端推理版)")
PROD_MAP = {"药用瓶口 (Bottle)": "bottle", "金属螺母 (Nut)": "metal_nut", "网格 (Grid)": "grid", "螺丝 (Screw)": "screw", "药片 (Pill)": "pill"}

# 确保难例库（反馈收集）文件夹存在
os.makedirs("feedback_negatives", exist_ok=True)

col_set, col_run = st.columns([1, 3])
with col_set:
    st.write("🔧 检测设置")
    selected = st.selectbox("检测对象", list(PROD_MAP.keys()))
    
    st.info("💡 **微服务架构提示**：当前前端已与 AI 推理核心解耦。图像将通过 HTTP POST 传至本地/云端 8000 端口的推理引擎。")
    
    mode = st.radio("数据来源", ["📁 本地上传", "📸 实时拍摄", "🖼️ 示例图库 (快速体验)"])
    sens = st.slider("画框敏感度", 90.0, 99.9, 99.5)
    
    # 允许在 UI 配置后端引擎的地址 (真实工程做法)
    api_host = st.text_input("后端引擎 API 地址", value="http://127.0.0.1:8000")

with col_run:
    uploaded_files = []
    
    # 数据源获取逻辑
    if mode == "🖼️ 示例图库 (快速体验)":
        st.info("💡 点击下方预置样本，一键体验微服务推理。")
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        if ex_col1.button("🔩 测试样本 1 (良品)"):
            uploaded_files = ["examples/sample_ok.jpg"] if os.path.exists("examples/sample_ok.jpg") else []
        if ex_col2.button("🔩 测试样本 2 (细微划痕)"):
            uploaded_files = ["examples/sample_ng1.jpg"] if os.path.exists("examples/sample_ng1.jpg") else []
        if ex_col3.button("🔩 测试样本 3 (严重变形)"):
            uploaded_files = ["examples/sample_ng2.jpg"] if os.path.exists("examples/sample_ng2.jpg") else []
            
        if not uploaded_files:
            st.warning("⚠️ 未找到示例图片，请在项目根目录创建 `examples` 文件夹并放入图片。")

    elif mode == "📁 本地上传":
        files = st.file_uploader("上传图片 (支持按住 Ctrl 多选)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
        if files: uploaded_files = files
    else:
        cam_file = st.camera_input("📸 工业视觉实时抓拍通道")
        if cam_file: uploaded_files = [cam_file]
        
    # ==========================================
    # 核心：调用 FastAPI 后端引擎
    # ==========================================
    if uploaded_files:
        st.success(f"✅ 任务建立！共收到 {len(uploaded_files)} 个检测任务，正在呼叫云端引擎...")
        
        for idx, file in enumerate(uploaded_files):
            file_name = file.name if hasattr(file, 'name') else str(file).split('/')[-1]
            st.markdown(f"#### 📄 样本 {idx + 1}: `{file_name}`")
            
            # 读取图片并重置尺寸供展示
            img = Image.open(file).convert('RGB')
            orig_res = cv2.resize(np.array(img), (256, 256))
            res_bgr = cv2.cvtColor(orig_res, cv2.COLOR_RGB2BGR)

            with st.spinner("🚀 正在将图像通过 RESTful API 发送至推理引擎..."):
                try:
                    # 1. 把图片转成二进制流，准备通过网络发送
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()

                    # 2. 组装请求 URL 并发送 HTTP POST 请求
                    api_url = f"{api_host}/api/v1/predict?selected_product={selected}&sens={sens}"
                    response = requests.post(api_url, files={"file": ("image.jpg", img_bytes, "image/jpeg")})
                    
                    # 3. 解析云端传回的检测结果
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get("status") == "success":
                            # 提取纯数据
                            mse = result["mse"]
                            max_err = result["max_err"]
                            defects = result["defects_count"]
                            bboxes = result["bboxes"]
                            err_map = np.array(result["err_map"]) # 还原热力图矩阵
                            
                            # 记录到履历数据库
                            log_to_db(selected, defects, mse)  
                            
                            # 前端仅负责“画框”，彻底告别张量计算
                            for box in bboxes:
                                x, y, w, h = box
                                cv2.rectangle(res_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)
                                
                            # ==========================================
                            # UI 渲染 1：算法置信度与严重级别
                            # ==========================================
                            st.markdown("##### 📊 算法置信度与严重级别")
                            if defects == 0:
                                severity_score = min(mse * 15000, 30) 
                                if severity_score < 20:
                                    st.success(f"🟢 **【品质优良】符合系统放行标准，准予流入封装工序。** | 异常指数: {severity_score:.1f}/100")
                                else:
                                    st.warning(f"🟡 **【轻度预警】检出临界边缘瑕疵，需流转至人工复核工位。** | 异常指数: {severity_score:.1f}/100")
                            else:
                                severity_score = min(60 + (defects * 8) + (max_err * 500), 100)
                                st.error(f"🔴 **锁定缺陷：{defects} 处。【严重违规】判定为不合格品 (NG)，自动触发剔除。** | 异常指数: {severity_score:.1f}/100")
                            st.progress(int(severity_score))
                            
                            # ==========================================
                            # UI 渲染 2：透视叠加分析 (Alpha Blending)
                            # ==========================================
                            st.markdown("##### 🔬 深度异常分析 (透视叠加)")
                            alpha = st.slider("拖动滑块调整热力图穿透层叠", min_value=0.0, max_value=1.0, value=0.6, step=0.05, key=f"alpha_slider_{idx}")
                            
                            heatmap_bgr = cv2.applyColorMap((err_map / (np.max(err_map)+1e-7) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                            orig_rgb = cv2.cvtColor(orig_res, cv2.COLOR_BGR2RGB)
                            
                            blended_img = cv2.addWeighted(orig_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
                            
                            col_a, col_b = st.columns([1, 1.5])
                            with col_a:
                                st.image(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB), caption="算法红框锁定 (BBox)", width="stretch")
                            with col_b:
                                st.image(blended_img, caption="像素级残差融合透视", width="stretch")
                            
                            # ==========================================
                            # UI 渲染 3：误判反馈收集
                            # ==========================================
                            col_fb1, col_fb2 = st.columns([3, 1])
                            with col_fb2:
                                if st.button("👎 误判反馈 (提交至难例库)", key=f"fb_btn_{idx}"):
                                    save_path = f"feedback_negatives/fb_{selected}_{idx}.jpg"
                                    img.save(save_path)
                                    st.toast("✅ 感谢反馈！已记录至主动学习待训练队列。", icon="📈")
                                    
                        else:
                            st.error(f"⚠️ 引擎返回错误: {result.get('message')}")
                    else:
                        st.error(f"⛔ 网络请求失败，状态码: {response.status_code}。请检查 API 参数！")
                        
                except requests.exceptions.ConnectionError:
                    st.error("🚨 **无法连接到推理引擎！**")
                    st.warning("请确保你已经打开了另一个终端窗口，并运行了 `uvicorn api_server:app --host 0.0.0.0 --port 8000`")
                except Exception as e:
                    st.error(f"发生未知错误: {str(e)}")
                    
            st.markdown("<hr style='border: 1px dashed #cbd5e1;'>", unsafe_allow_html=True)