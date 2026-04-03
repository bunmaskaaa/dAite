FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first, then everything else
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt --no-deps sentence-transformers
RUN pip install --no-cache-dir faiss-cpu fastapi uvicorn numpy pandas \
    transformers tokenizers huggingface-hub safetensors \
    scikit-learn tqdm pydantic pyyaml regex filelock \
    requests packaging typing-extensions

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]