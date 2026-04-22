from fastapi import FastAPI, File, UploadFile
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import io
import uvicorn
# 假设你的核心加载代码还在 core_utils 里
from core_utils import load_model, DEVICE 

app = FastAPI(title="CAE-BN 视觉推理引擎 API", version="1.0")

PROD_MAP = {"药用瓶口 (Bottle)": "bottle", "金属螺母 (Nut)": "metal_nut", "网格 (Grid)": "grid", "螺丝 (Screw)": "screw", "药片 (Pill)": "pill"}
preprocess = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# 提前加载好所有模型到内存字典中，避免每次请求都重新加载 (工业级性能优化)
models = {}
for prod_name, file_prefix in PROD_MAP.items():
    try:
        models[prod_name] = load_model(f"weights/{file_prefix}_ae.pth")
    except Exception as e:
        print(f"警告: 模型 {file_prefix}_ae.pth 未找到。")

@app.post("/api/v1/predict")
async def predict_defect(selected_product: str, sens: float = 99.5, file: UploadFile = File(...)):
    """
    核心视觉推理接口：接收前端传来的图片，返回 MSE 误差和缺陷坐标
    """
    if selected_product not in models or models[selected_product] is None:
        return {"status": "error", "message": "该产品的模型尚未加载"}
    
    # 1. 读取前端传来的二进制图片
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # 2. AI 推理
    model = models[selected_product]
    t = preprocess(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(t)
        
    # 3. 计算残差与缺陷位置
    err_map = torch.mean(torch.pow(t - output, 2), dim=1).squeeze().cpu().numpy()
    mse = float(np.mean(err_map))
    max_err = float(np.max(err_map))
    
    mask = cv2.morphologyEx(np.where(err_map > np.percentile(err_map, sens), 255, 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 只返回画框所需的坐标数据，不传回笨重的图片
    bboxes = []
    for c in contours:
        if cv2.contourArea(c) > 20:
            x, y, w, h = cv2.boundingRect(c)
            bboxes.append([int(x), int(y), int(w), int(h)])
            
    return {
        "status": "success",
        "mse": mse,
        "max_err": max_err,
        "defects_count": len(bboxes),
        "bboxes": bboxes, # 将坐标传给前端，让前端自己去画红框
        # 为了极简，我们将 err_map 转为列表传回用于热力图 (实际工业中会转成 base64 图像传回)
        "err_map": err_map.tolist() 
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)