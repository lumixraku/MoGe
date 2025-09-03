# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoGe (Monocular Geometry Estimation) is a deep learning model for recovering 3D geometry from monocular open-domain images. It can estimate metric point maps, metric depth maps, normal maps, and camera FOV from single images.

The repository contains two major versions:
- MoGe-1: Original model for geometry estimation
- MoGe-2: Improved model with metric scale and sharp details

## Repository Structure

```
moge/
├── model/          # Core model implementations (v1.py, v2.py)
├── scripts/        # Command-line tools and applications
├── utils/          # Utility functions for geometry, IO, visualization
├── train/          # Training scripts and utilities
├── test/           # Test scripts
└── dinov2/         # DINOv2 backbone implementation
```

## Key Commands

### Installation
```bash
pip install -r requirements.txt
# or
pip install git+https://github.com/microsoft/MoGe.git
```

### Running the demo application
```bash
moge app
# or
python moge/scripts/app.py
```

### Inference on images
```bash
moge infer -i INPUT_IMAGE_OR_FOLDER --o OUTPUT_FOLDER --maps --glb --ply
```

### Running tests
```bash
# Tests are typically run with pytest
python -m pytest moge/test/
```

## Model Architecture

The model is built on a Vision Transformer (ViT) backbone with custom modules for geometry estimation. Key components:

- `moge/model/v1.py` and `moge/model/v2.py`: Main model implementations
- `moge/model/modules.py`: Custom neural network modules
- `moge/model/dinov2/`: DINOv2 vision transformer backbone

## Development Workflow

1. Models are implemented in `moge/model/` with separate files for v1 and v2
2. Command-line tools are in `moge/scripts/` 
3. Utilities are in `moge/utils/`
4. Training code is in `moge/train/`
5. Tests are in `moge/test/`

## Key Dependencies

- PyTorch >= 2.0.0
- OpenCV
- HuggingFace Hub
- Gradio (for demo)
- Trimesh (for 3D processing)
- Utils3D (custom geometry utilities)

## Entry Points

- CLI: `moge/scripts/cli.py`
- Web demo: `moge/scripts/app.py`
- Inference: `moge/scripts/infer.py`