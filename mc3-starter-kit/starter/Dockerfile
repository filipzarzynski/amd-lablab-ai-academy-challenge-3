# Mandated ROCm base image for AMD AI League Match 3.
# Checked by layer identity before container run.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Ensure non-destructive dependency installation to protect ROCm PyTorch
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -U transformers tokenizers safetensors

# Verify ROCm PyTorch was not replaced by CUDA PyTorch
RUN python3 -c "import torch; assert '+rocm' in torch.__version__, f'ROCm torch corrupted: {torch.__version__}'"

# If bundling local model weights for offline evaluation:
# COPY models/ /models/

COPY app/ /app/

RUN mkdir -p /app/corpus /app/output /app/index

# Long-running daemon keeps model and index warm in GPU VRAM
CMD ["python3", "/app/server.py"]
