# Use a PyTorch base image without CUDA
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
      apt-get update && \
      DEBIAN_FRONTEND=noninteractive apt-get install -y \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1-mesa-glx \
        tzdata \
        && rm -rf /var/lib/apt/lists/*

# Install uv for faster dependency management
RUN pip install uv

# Ensure uv is available in the runtime environment
ENV PATH="/usr/local/bin:$PATH"

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN uv venv && \
    uv pip install -r requirements.txt

# Copy source code
COPY . .

# Expose port for FastAPI app
EXPOSE 8000

# Set environment variable for OpenCV
ENV OPENCV_IO_ENABLE_OPENEXR=1

# Run the FastAPI app
CMD ["/bin/bash", "-c", "uv venv && uv sync && uv run python simple_api.py"]