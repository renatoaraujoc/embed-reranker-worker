FROM --platform=linux/amd64 nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ── Python 3.11 + pip ───────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 git && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    rm -rf /var/lib/apt/lists/*

# ── 1. torch with CUDA 12.4 ────────────────────────────────────────────
RUN python3 -m pip install --no-cache-dir \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# ── 2. Application dependencies (cached until requirements.txt changes) ─
COPY requirements.txt /app/
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# ── 3. flash-attn prebuilt wheel (cxx11abiFALSE matches pip-installed torch)
RUN python3 -m pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# ── 4. Source code (changes most often → last layer) ────────────────────
COPY src/ /app/src/

WORKDIR /app/src
CMD ["python3", "handler.py"]
