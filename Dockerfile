FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git curl wget && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY upstream/pyproject.toml ./upstream/pyproject.toml
COPY upstream/uv.lock* ./upstream/
COPY upstream/prepare.py ./upstream/
COPY upstream/train.py ./upstream/
COPY upstream/program.md ./upstream/
COPY patches/ ./patches/
COPY configs/ ./configs/
COPY programs/ ./programs/
COPY dashboard/ ./dashboard/
COPY entrypoint.sh ./

# Fix Windows CRLF line endings
RUN sed -i 's/\r$//' entrypoint.sh && sed -i 's/\r$//' patches/*.py

# Install dependencies via pip (more reliable in Docker than uv sync)
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu124 && \
    python -m pip install --no-cache-dir \
    matplotlib numpy pandas pyarrow requests rustbpe tiktoken && \
    python -m pip install --no-cache-dir datasets || true

# Make entrypoint executable
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
