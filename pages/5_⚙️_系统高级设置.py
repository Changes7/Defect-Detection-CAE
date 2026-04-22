import streamlit as st
import time
import os
import shutil
import datetime

# 第一重锁：必须登录
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    st.error("🛑 拒绝访问：您尚未登录系统。")
    st.stop()

# 第二重锁：必须是管理员 (RBAC 核心)
if st.session_state.role != "管理员":
    st.error("⛔ 越权拦截：您的当前角色为【操作员】，无权访问系统底层配置模块！")
    st.image("https://http.cat/403", width=400) # 调皮一下，显示 403 没权限的猫咪图
    st.stop() # 代码在这里被强制掐断，下面的设置界面根本不会渲染出来


st.set_page_config(page_title="系统高级设置", page_icon="⚙️", layout="wide")

st.title("⚙️ 系统高级设置与边缘控制 (Advanced Settings)")
st.markdown("⚠️ **管理员权限专属**：在此模块调整的全局参数将直接影响底层视觉引擎的推理逻辑与外设联动状态。请谨慎操作。")
st.markdown("---")

# 初始化 session_state 来保存模拟的设置状态
if 'settings_saved' not in st.session_state:
    st.session_state.settings_saved = False

# ==========================================
# 1. 算法核心参数配置
# ==========================================
st.header("🧠 1. 视觉算法全局参数 (Algorithm Configuration)")
col1, col2 = st.columns(2)

with col1:
    st.subheader("异常判定基线 (Anomaly Threshold)")
    default_sens = st.slider("全局默认 MSE 敏感度 (Percentile)", min_value=90.0, max_value=99.9, value=99.5, step=0.1, help="数值越高，系统越严格，越容易将微小瑕疵判定为不合格。")
    ssim_weight = st.slider("SSIM 结构相似度权重分配", min_value=0.0, max_value=1.0, value=0.3, step=0.1, help="结合 SSIM 与 MSE 的复合损失判定，增强对抗光照变化的鲁棒性。")

with col2:
    st.subheader("推理引擎设定 (Inference Engine)")
    engine_mode = st.selectbox("边缘端推理加速方案", ["原生 PyTorch (CPU/CUDA)", "ONNX Runtime (跨平台)", "TensorRT (极致性能)", "OpenVINO (Intel 优化)"])
    st.toggle("开启 FP16 半精度推理 (降低显存占用，提速 40%)", value=True)

st.markdown("---")

# ==========================================
# 2. 硬件与外设通道映射
# ==========================================
st.header("🔌 2. 外设通信与 IO 控制 (Hardware Integration)")
col3, col4 = st.columns(2)

with col3:
    st.subheader("视频流通道分配 (Camera Channel)")
    cam_index = st.selectbox("默认采集设备映射", ["摄像头 0 (默认笔记本/USB)", "摄像头 1 (外接高帧率)", "摄像头 2 (工业 GigE 接口)", "RTSP 网络推流协议"])
    if cam_index == "RTSP 网络推流协议":
        st.text_input("RTSP 流地址 (例如: rtsp://admin:12345@192.168.1.64/554)")

with col4:
    st.subheader("PLC 自动化联动 (Automation IO)")
    st.toggle("开启 Modbus TCP 协议通信", value=False, help="开启后，检测出【不合格】时将向 PLC 发送剔除指令。")
    st.text_input("下位机 PLC IP 地址", value="192.168.0.100")
    st.toggle("驱动声光报警器 (继电器 COM 口)", value=True)

st.markdown("---")

# ==========================================
# 3. 数据安全与维护
# ==========================================
st.header("🛡️ 3. 数据驻留与系统维护 (Maintenance)")
col5, col6 = st.columns(2)

with col5:
    st.subheader("履历数据库备份")
    st.write("定期备份 `defect.db` 以防数据丢失。")
    if st.button("💾 执行全量数据库备份"):
        if os.path.exists("defect.db"):
            backup_name = f"defect_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy("defect.db", backup_name)
            st.success(f"✅ 备份成功！已生成安全快照：`{backup_name}`")
        else:
            st.error("❌ 找不到原始数据库 `defect.db`，请先进行质检测试。")

with col6:
    st.subheader("缓存清理机制")
    st.write("释放 Streamlit 页面缓存与 PyTorch 显存。")
    if st.button("🧹 一键清理系统垃圾"):
        with st.spinner("正在强制回收内存..."):
            time.sleep(1.5)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("✅ 缓存已清空，系统性能已恢复至最佳状态。")

st.markdown("---")

# ==========================================
# 4. 底部保存操作区
# ==========================================
c_left, c_mid, c_right = st.columns([1, 2, 1])
with c_mid:
    if st.button("🔄 应用并保存所有全局配置", type="primary", use_container_width=True):
        with st.spinner("正在向底层引擎下发配置参数..."):
            time.sleep(1.2)  # 模拟配置下发延迟
            st.toast("✅ 配置下发成功！系统已热更新。", icon="🚀")
            st.session_state.settings_saved = True

if st.session_state.settings_saved:
    st.info("💡 提示：新的推理加速方案和阈值将在下一次【AI 质检终端】运行时生效。")