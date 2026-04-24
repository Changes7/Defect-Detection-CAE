import streamlit as st
import streamlit.components.v1 as components
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from core_utils import load_model, DEVICE
from style_utils import apply_glass_theme
apply_glass_theme()

st.title("🧬 核心算法原理与深度解析")

# ==========================================
# 模块 1：前端动态交互演示 (嵌入优化的 HTML/JS)
# ==========================================
st.header("🎛️ 1. CAE 数据流沙漏模型 (交互演示)")
st.write("拖动下方滑块，直观感受工业高维图像数据在深度神经网络中的“降维压缩”与“升维重构”全过程。")

cae_animation_html = """
<!DOCTYPE html>
<html>
<head>
<style>
  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; overflow: hidden; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 10px;}
  .container { display: flex; justify-content: center; align-items: center; height: 260px; gap: 8px; margin-top: 10px;}
  .layer { 
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    background: #e2e8f0; border: 2px solid #cbd5e1; border-radius: 8px; transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 6px rgba(0,0,0,0.05); position: relative; opacity: 0.6;
  }
  .layer.active { 
    background: linear-gradient(135deg, #3b82f6, #2563eb); border-color: #1d4ed8; 
    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.4); opacity: 1; transform: scale(1.15); z-index: 10;
  }
  
  /* 沙漏形状比例 */
  #l0 { height: 220px; width: 45px; }  
  #l1 { height: 160px; width: 65px; }  
  #l2 { height: 100px; width: 85px; }  
  #l3 { height: 60px;  width: 130px; } /* Bottleneck */
  #l4 { height: 100px; width: 85px; }  
  #l5 { height: 160px; width: 65px; }  
  #l6 { height: 220px; width: 45px; }  
  
  .label { font-size: 13px; margin-top: 12px; color: #475569; font-weight: 600;}
  .active + .label { color: #2563eb; font-weight: 800; }
  
  .slider-container { margin: 20px auto; width: 85%; }
  input[type=range] { width: 100%; cursor: pointer; accent-color: #2563eb; }
  
  #info-box { 
    margin-top: 20px; padding: 25px; background-color: #ffffff; border-left: 6px solid #3b82f6;
    border-radius: 8px; text-align: left; max-width: 850px; margin-left: auto; margin-right: auto;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
  }
  h2 { margin-top: 0; color: #1e293b; font-size: 20px;}
  .dim-text { color: #d97706; font-size: 1.1em; font-family: monospace; font-weight: bold; background: #fef3c7; padding: 3px 8px; border-radius: 4px;}
  p { color: #475569; line-height: 1.6; font-size: 15px;}
</style>
</head>
<body>
  <div class="slider-container">
    <input type="range" id="stageSlider" min="0" max="6" value="0" oninput="updateStage(this.value)">
  </div>

  <div class="container">
    <div><div class="layer active" id="l0"></div><div class="label">Input</div></div>
    <div><div class="layer" id="l1"></div><div class="label">Enc 1</div></div>
    <div><div class="layer" id="l2"></div><div class="label">Enc 2</div></div>
    <div><div class="layer" id="l3"></div><div class="label">Bottleneck</div></div>
    <div><div class="layer" id="l4"></div><div class="label">Dec 1</div></div>
    <div><div class="layer" id="l5"></div><div class="label">Dec 2</div></div>
    <div><div class="layer" id="l6"></div><div class="label">Output</div></div>
  </div>

  <div id="info-box">
    <h2 id="stage-title">阶段 0: 原始输入图像 (Input)</h2>
    <p><strong>张量维度 (C × H × W)：</strong> <span id="stage-dim" class="dim-text">3 × 256 × 256</span></p>
    <p id="stage-desc">原始的工业产品 RGB 彩色图像。由于分辨率较高，它包含了丰富的局部纹理和背景噪音信息，但还没有被提取出高级语义特征。</p>
  </div>

  <script>
    const data = [
      { title: "阶段 0: 原始输入图像 (Input)", dim: "3 × 256 × 256", desc: "原始的工业产品 RGB 彩色图像。由于分辨率较高，它包含了丰富的局部纹理和背景噪音信息，但还没有被提取出高级语义特征。" },
      { title: "阶段 1: 第一层特征编码 (Encoder 1)", dim: "32 × 128 × 128", desc: "经过步长为 2 的卷积层处理，图像的长宽尺寸缩小一半，同时通道数（提取的特征种类）扩展到了 32 维。模型开始识别边缘和基础纹理。" },
      { title: "阶段 2: 第二层特征编码 (Encoder 2)", dim: "64 × 64 × 64", desc: "空间分辨率继续减半，通道数加深至 64。数据逐渐由具体的图像像素转化为抽象的特征矩阵。" },
      { title: "阶段 3: 极致压缩瓶颈层 (Bottleneck)", dim: "128 × 32 × 32", desc: "网络的最核心区域！空间尺寸被极度压缩，强制丢弃细微的瑕疵和噪音，通道数达到最深的 128，只保留代表产品正常结构的最高级抽象特征。" },
      { title: "阶段 4: 第一层特征解码 (Decoder 1)", dim: "64 × 64 × 64", desc: "开始重构过程。通过转置卷积（反卷积），模型尝试将高维特征放大，恢复空间分辨率。" },
      { title: "阶段 5: 第二层特征解码 (Decoder 2)", dim: "32 × 128 × 128", desc: "尺寸进一步放大至 128x128，开始还原出产品的具体形状和轮廓细节。" },
      { title: "阶段 6: 最终重构输出 (Output)", dim: "3 × 256 × 256", desc: "完美复原回 256x256 的尺寸和 3 个色彩通道。由于瓶颈层过滤了瑕疵信息，此处生成的图像将是一张【没有任何缺陷】的完美产品图。" }
    ];

    function updateStage(val) {
      for(let i=0; i<=6; i++) {
        document.getElementById('l'+i).className = (i == val) ? 'layer active' : 'layer';
      }
      document.getElementById('stage-title').innerText = data[val].title;
      document.getElementById('stage-dim').innerText = data[val].dim;
      document.getElementById('stage-desc').innerText = data[val].desc;
    }
  </script>
</body>
</html>
"""
components.html(cae_animation_html, height=520)

st.markdown("---")

# ==========================================
# 模块 2：真实模型代码剖析与实时推理
# ==========================================
st.header("⚙️ 2. 真实物理模型解剖 (Live Forward Pass)")
st.write("理论结合实际：在这里，你可以直接透视项目中训练好的 PyTorch 真实权重，并输入图片验证上述的“降维与重构”理论。")

PROD_MAP = {"药用瓶口 (Bottle)": "bottle", "金属螺母 (Nut)": "metal_nut", "网格 (Grid)": "grid", "螺丝 (Screw)": "screw", "药片 (Pill)": "pill"}
selected_prod = st.selectbox("1️⃣ 选择要探究的底层模型权重", list(PROD_MAP.keys()))

model_path = f"weights/{PROD_MAP[selected_prod]}_ae.pth"
model = load_model(model_path)

if model is None:
    st.error(f"❌ 找不到模型权重: {model_path}，请检查 weights 文件夹。")
else:
    with st.expander("👁️ 展开查看底层网络拓扑结构代码 (PyTorch Architecture)", expanded=False):
        st.code(str(model), language='python')
        st.caption(f"当前运行设备: {DEVICE}")

    file = st.file_uploader("2️⃣ 上传一张图片，运行真实的数学前向传播 (Forward)", type=['png', 'jpg', 'jpeg'])
    
    if file:
        img = Image.open(file).convert('RGB')
        preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        input_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        st.markdown("### 🔀 逐层推演结果")
        col1, col2, col3 = st.columns([1.2, 1, 1.2])

        with col1:
            st.markdown("**[1] 真实输入 (Tensor)**")
            st.image(img, use_container_width=True)
            st.info(f"📐 维度:\n`{list(input_tensor.shape)}`")

        with col2:
            st.markdown("**[2] 空间压缩特征**")
            st.markdown("<br>", unsafe_allow_html=True)
            st.warning("⬇️ 经过 Encoder 网络极度压缩\n\n提取核心工业纹理，物理丢弃表面细微划痕。")
            st.markdown("<br>", unsafe_allow_html=True)

        with col3:
            st.markdown("**[3] 物理重构输出**")
            with torch.no_grad():
                output_tensor = model(input_tensor)
            
            recon_img = np.clip(output_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
            st.image(recon_img, use_container_width=True)
            st.success(f"⬆️ 经过 Decoder 重构\n\n📐 维度:\n`{list(output_tensor.shape)}`")

        st.markdown("### 🧮 算法核心：MSE 残差评估")
        err_map = torch.mean(torch.pow(input_tensor - output_tensor, 2), dim=1).squeeze().cpu().numpy()
        heatmap = cv2.applyColorMap((err_map / (np.max(err_map)+1e-7) * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), caption="MSE 残差热力图 (颜色越偏红，存在缺陷的数学概率越大)", use_container_width=True)
        with c2:
            st.markdown("<br>", unsafe_allow_html=True)
            mse_value = float(np.mean(err_map))
            st.metric("全局均方误差 (MSE)", f"{mse_value:.6f}")
            st.caption("提示：在检测工位中，系统正是根据此热力图的像素阈值来绘制红色缺陷警告框的。")