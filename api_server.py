#!/usr/bin/env python3
"""
FastAPI server for MoGe depth estimation API
Provides REST API endpoints for depth estimation and 3D reconstruction
"""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import sys
from pathlib import Path
if (_package_root := str(Path(__file__).absolute().parents[0])) not in sys.path:
    sys.path.insert(0, _package_root)

import io
import base64
import tempfile
import uuid
from typing import Dict, Optional, List
import json

import cv2
import torch
import numpy as np
import trimesh
import trimesh.visual
from PIL import Image
import utils3d
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from moge.utils.io import write_normal
from moge.utils.vis import colorize_depth, colorize_normal
from moge.model import import_model_class_by_version
from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
from moge.utils.tools import timeit

app = FastAPI(
    title="MoGe Depth Estimation API",
    description="API for depth estimation and 3D reconstruction using MoGe model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
model_version = None
use_fp16 = False

def load_model(pretrained_model_name_or_path: str = None, version: str = "v2", fp16: bool = False):
    """Load the MoGe model"""
    global model, model_version, use_fp16

    print("Loading model...")
    if pretrained_model_name_or_path is None:
        DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
            "v1": "Ruicheng/moge-vitl",
            "v2": "Ruicheng/moge-2-vitl-normal",
        }
        pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION[version]

    model = import_model_class_by_version(version).from_pretrained(pretrained_model_name_or_path).cuda().eval()
    if fp16:
        model.half()

    model_version = version
    use_fp16 = fp16
    print(f"Model loaded: {pretrained_model_name_or_path} (version: {version}, fp16: {fp16})")

def run_inference(image: np.ndarray, resolution_level: int, apply_mask: bool) -> Dict[str, np.ndarray]:
    """Run inference on GPU"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    image_tensor = torch.tensor(image, dtype=torch.float32 if not use_fp16 else torch.float16, device=torch.device('cuda')).permute(2, 0, 1) / 255
    output = model.infer(image_tensor, apply_mask=apply_mask, resolution_level=resolution_level, use_fp16=use_fp16)
    output = {k: v.cpu().numpy() for k, v in output.items()}
    return output

def process_image(
    image: np.ndarray,
    max_size: int = 800,
    resolution_level: str = 'High',
    apply_mask: bool = True,
    remove_edge: bool = True
):
    """Full inference pipeline"""
    # Resize image if needed
    larger_size = max(image.shape[:2])
    if larger_size > max_size:
        scale = max_size / larger_size
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    height, width = image.shape[:2]

    # Convert resolution level to int
    resolution_level_int = {'Low': 0, 'Medium': 5, 'High': 9, 'Ultra': 30}.get(resolution_level, 9)

    # Run inference
    output = run_inference(image, resolution_level_int, apply_mask)

    points, depth, mask, normal = output['points'], output['depth'], output['mask'], output.get('normal', None)

    if remove_edge:
        mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=0.04)
    else:
        mask_cleaned = mask

    results = {
        **output,
        'mask_cleaned': mask_cleaned,
        'image': image
    }

    # Calculate FOV
    intrinsics = results['intrinsics']
    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
    fov_x, fov_y = np.rad2deg([fov_x, fov_y])

    results['fov_x'] = float(fov_x)
    results['fov_y'] = float(fov_y)

    return results

def numpy_to_base64(arr: np.ndarray) -> str:
    """Convert numpy array to base64 string"""
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # For depth/normal maps, convert to uint8 for visualization
        if len(arr.shape) == 2:  # depth map
            arr_vis = colorize_depth(arr)
        else:  # normal map
            arr_vis = colorize_normal(arr)
        arr = arr_vis

    # Convert to PIL Image and then to base64
    if len(arr.shape) == 3:
        img = Image.fromarray(arr.astype(np.uint8))
    else:
        img = Image.fromarray(arr.astype(np.uint8))

    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MoGe Depth Estimation API",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "model_version": model_version,
        "endpoints": {
            "/predict": "POST - Upload image for depth estimation",
            "/predict_base64": "POST - Send base64 image for depth estimation",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_version": model_version
    }

@app.post("/predict")
async def predict_depth(
    file: UploadFile = File(...),
    max_size: int = Form(800),
    resolution_level: str = Form("High"),
    apply_mask: bool = Form(True),
    remove_edge: bool = Form(True),
    return_format: str = Form("json")  # json, files, base64
):
    """
    Predict depth from uploaded image file

    Parameters:
    - file: Image file (jpg, png, etc.)
    - max_size: Maximum image size (default: 800)
    - resolution_level: Low, Medium, High, Ultra (default: High)
    - apply_mask: Apply mask to output (default: True)
    - remove_edge: Remove edge artifacts (default: True)
    - return_format: Response format - json, files, base64 (default: json)
    """
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process image
        results = process_image(image, max_size, resolution_level, apply_mask, remove_edge)

        if return_format == "base64":
            # Return base64 encoded images
            response = {
                "depth_map": numpy_to_base64(results['depth']),
                "fov_x": results['fov_x'],
                "fov_y": results['fov_y'],
                "image_shape": list(image.shape)
            }

            if 'normal' in results and results['normal'] is not None:
                response["normal_map"] = numpy_to_base64(results['normal'])

            return JSONResponse(content=response)

        elif return_format == "json":
            # Return JSON with basic info
            response = {
                "success": True,
                "image_shape": list(image.shape),
                "depth_shape": list(results['depth'].shape),
                "fov_x": results['fov_x'],
                "fov_y": results['fov_y'],
                "has_normal": 'normal' in results and results['normal'] is not None,
                "message": "Depth estimation completed successfully"
            }

            if 'normal' in results and results['normal'] is not None:
                response["normal_shape"] = list(results['normal'].shape)

            return JSONResponse(content=response)

        else:
            return JSONResponse(content={"error": "Unsupported return_format"}, status_code=400)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict_base64")
async def predict_depth_base64(
    image_data: dict,
    max_size: int = 800,
    resolution_level: str = "High",
    apply_mask: bool = True,
    remove_edge: bool = True,
    return_format: str = "json"
):
    """
    Predict depth from base64 encoded image

    Body should contain:
    {
        "image": "base64_encoded_image_string",
        "max_size": 800,
        "resolution_level": "High",
        "apply_mask": true,
        "remove_edge": true,
        "return_format": "json"
    }
    """
    try:
        # Decode base64 image
        image_b64 = image_data.get("image", "")
        if not image_b64:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Remove data URL prefix if present
        if image_b64.startswith('data:image'):
            image_b64 = image_b64.split(',')[1]

        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get parameters from request body
        max_size = image_data.get("max_size", max_size)
        resolution_level = image_data.get("resolution_level", resolution_level)
        apply_mask = image_data.get("apply_mask", apply_mask)
        remove_edge = image_data.get("remove_edge", remove_edge)
        return_format = image_data.get("return_format", return_format)

        # Process image
        results = process_image(image, max_size, resolution_level, apply_mask, remove_edge)

        if return_format == "base64":
            # Return base64 encoded images
            response = {
                "depth_map": numpy_to_base64(results['depth']),
                "fov_x": results['fov_x'],
                "fov_y": results['fov_y'],
                "image_shape": list(image.shape)
            }

            if 'normal' in results and results['normal'] is not None:
                response["normal_map"] = numpy_to_base64(results['normal'])

            return JSONResponse(content=response)

        elif return_format == "json":
            # Return JSON with basic info
            response = {
                "success": True,
                "image_shape": list(image.shape),
                "depth_shape": list(results['depth'].shape),
                "fov_x": results['fov_x'],
                "fov_y": results['fov_y'],
                "has_normal": 'normal' in results and results['normal'] is not None,
                "message": "Depth estimation completed successfully"
            }

            if 'normal' in results and results['normal'] is not None:
                response["normal_shape"] = list(results['normal'].shape)

            return JSONResponse(content=response)

        else:
            return JSONResponse(content={"error": "Unsupported return_format"}, status_code=400)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoGe API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--pretrained", default=None, help="Pretrained model path")
    parser.add_argument("--version", default="v2", help="Model version")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 inference")

    args = parser.parse_args()

    # Load model with specified parameters
    load_model(args.pretrained, args.version, args.fp16)

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)
