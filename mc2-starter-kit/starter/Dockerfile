# Mandated ROCm base image for AMD AI League Match 2.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -U transformers tokenizers safetensors

# Verify ROCm PyTorch was not replaced by CUDA PyTorch
RUN python3 -c "import torch; assert '+rocm' in torch.__version__, f'ROCm torch corrupted: {torch.__version__}'"

# If bundling local model weights:
# COPY models/ /models/

COPY app/ /app/

RUN mkdir -p /app/input /app/output

# Long-running daemon keeps OCR model warm in GPU VRAM
CMD ["python3", "/app/server.py"]
