FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    git+https://github.com/coqui-ai/TTS \
    deepspeed \
    transformers \
    huggingface_hub \
    runpod

# Initialize git-lfs
RUN git lfs install

# Download EGTTS model
RUN git clone https://huggingface.co/OmarSamir/EGTTS-V0.1 /models/EGTTS-V0.1

# Copy all files
COPY . /app/

# Set environment
ENV COQUI_TOS_AGREED=1

# Run handler
CMD ["python", "-u", "/app/handler.py"]