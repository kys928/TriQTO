FROM python:3.11-slim-bookworm

ARG TRIQTO_TORCH_PROFILE=cuda
ARG TRIQTO_TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu128

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRIQTO_TORCH_PROFILE=${TRIQTO_TORCH_PROFILE} \
    TRIQTO_TORCH_CUDA_INDEX=${TRIQTO_TORCH_CUDA_INDEX}

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/TriQTO
COPY . .
RUN bash docker/phase15_6/bootstrap_pod.sh /workspace/TriQTO

VOLUME ["/workspace/triqto-data"]
CMD ["bash"]
