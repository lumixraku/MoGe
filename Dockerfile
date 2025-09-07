# Use a standard Python base image
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies (标准镜像已包含 build-essential)
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      libgomp1 \
      libgl1 \
      libgl1-mesa-dev \
      libx11-dev \
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