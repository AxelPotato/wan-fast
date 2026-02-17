# -----------------------------------------------------------------------------
# Stage 1: Builder (Compiler Environment)
# -----------------------------------------------------------------------------
FROM vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py310 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"

# Install System Build Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ninja-build \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.8 (Required for SM_120 / Blackwell Support)
RUN pip3 install --no-cache-dir --retries 5 --timeout 120 \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Compile SageAttention 2 from Source
# TORCH_CUDA_ARCH_LIST is required since no GPU is available during docker build
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN pip3 install --no-cache-dir --upgrade pip packaging setuptools wheel \
    && git clone https://github.com/thu-ml/SageAttention.git \
    && cd SageAttention \
    && MAX_JOBS=4 pip3 install --no-cache-dir --no-build-isolation .

# -----------------------------------------------------------------------------
# Stage 2: Runtime (Production Environment)
# -----------------------------------------------------------------------------
FROM vastai/base-image:cuda-12.8.1-cudnn-devel-ubuntu22.04-py310

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Runtime Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Pre-Built Python Packages from Builder Stage
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install Application-Specific Libraries
# Diffusers must be updated to support Wan 2.2
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    "diffusers>=0.33.0" \
    transformers \
    accelerate \
    sentencepiece \
    protobuf \
    huggingface_hub

# Copy Application Code
WORKDIR /workspace
COPY . .
RUN chmod +x entrypoint.sh

# Expose the API Port
EXPOSE 8000

# Start the Service
CMD ["./entrypoint.sh"]