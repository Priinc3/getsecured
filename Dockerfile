FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# ponytail: CPU-only torch first so ultralytics doesn't drag 7GB of CUDA wheels into a 512MB-RAM build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
# ponytail: serial dlib compile — parallel make OOM-kills small builders
RUN CMAKE_BUILD_PARALLEL_LEVEL=1 pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "streamlit run src/app.py --server.port=${PORT:-7860} --server.address=0.0.0.0 --server.headless=true"]
