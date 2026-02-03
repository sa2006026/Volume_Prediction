# Dockerfile for SAM Segmentation Flask App (GPU version)
# Uses PyTorch base image with CUDA support

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set timezone non-interactively to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Note: PyTorch is already installed in base image with CUDA support
RUN pip install --no-cache-dir -r requirements.txt

# Verify PyTorch CUDA is available
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads results/sam_segmentation templates model

# Expose Flask port
EXPOSE 5013

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=src/web/sam_website.py
ENV PYTHONPATH=/app

# Run the application
CMD ["python", "src/web/sam_website.py"]
