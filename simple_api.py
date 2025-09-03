#!/usr/bin/env python3
"""
超简单的 MoGe API - 直接传图片，直接返回结果
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[0])) not in sys.path:
    sys.path.insert(0, _package_root)

import json
import cv2
import torch
import numpy as np
from PIL import Image
import utils3d
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from moge.utils.vis import colorize_depth, colorize_normal
from moge.model import import_model_class_by_version

app = FastAPI()

# 全局模型
model = None

def load_model():
    global model
    print("加载模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    model = import_model_class_by_version("v2").from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
    print("模型加载完成!")

@app.on_event("startup")
async def startup():
    load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """上传图片，返回深度估计结果"""
    try:
        # 读取图片
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整图片大小
        larger_size = max(image.shape[:2])
        if larger_size > 800:
            scale = 800 / larger_size
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        # 推理 - 使用与命令行相同的参数
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        use_fp16 = device == 'cuda'  # CPU 不支持 fp16
        image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
        output = model.infer(image_tensor, resolution_level=9, use_fp16=use_fp16)

        # 生成深度图
        depth = output['depth'].cpu().numpy()
        depth_vis = colorize_depth(depth)

        # 保存到 output 目录
        os.makedirs("./output", exist_ok=True)

        # 获取文件名（不含扩展名）
        filename = os.path.splitext(file.filename)[0]

        # 保存深度图
        depth_path = f"./output/{filename}_depth.png"
        cv2.imwrite(depth_path, cv2.cvtColor(depth_vis, cv2.COLOR_RGB2BGR))

        # 保存原始深度数据
        depth_raw_path = f"./output/{filename}_depth.exr"
        cv2.imwrite(depth_raw_path, depth.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])

        # 如果有法线图，也保存
        if 'normal' in output and output['normal'] is not None:
            normal = output['normal'].cpu().numpy()
            normal_vis = colorize_normal(normal)
            normal_path = f"./output/{filename}_normal.png"
            cv2.imwrite(normal_path, cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR))

        # 计算FOV
        intrinsics = output['intrinsics'].cpu().numpy()
        fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
        fov_x, fov_y = np.rad2deg([fov_x, fov_y])

        # 准备响应信息
        fov_info = {
            "fov_x": float(fov_x),
            "fov_y": float(fov_y),
            "image_shape": list(image.shape),
            "depth_shape": list(depth.shape),
            "files_saved": [depth_path, depth_raw_path]
        }

        if 'normal' in output and output['normal'] is not None:
            fov_info["files_saved"].append(normal_path)

        # 保存FOV信息到文件
        with open(f"./output/{filename}_info.json", 'w') as f:
            json.dump(fov_info, f, indent=2)

        # 返回FOV信息作为JSON响应
        return JSONResponse(content=fov_info)

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """下载生成的文件"""
    file_path = f"./output/{filename}"
    
    if not os.path.exists(file_path):
        return {"error": f"File {filename} not found"}
    
    # 根据文件扩展名设置 media_type
    if filename.endswith('.png'):
        media_type = 'image/png'
    elif filename.endswith('.exr'):
        media_type = 'application/octet-stream'
    elif filename.endswith('.json'):
        media_type = 'application/json'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(file_path, media_type=media_type, filename=filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
