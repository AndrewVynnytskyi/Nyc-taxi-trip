# syntax=docker/dockerfile:1.7
# ==============================================================================
# NYC Taxi Trip Analytics -- production image.
#
# Multi-stage build:
#   1. `builder`  -- compiles wheels into an isolated virtualenv (build tools are
#                    never shipped to production).
#   2. `runtime`  -- slim CUDA-enabled base, non-root user, virtualenv copied in.
#
# Build (GPU, default):
#   docker build -t nyc-taxi:latest .
# Build (CPU only, no CUDA runtime layer):
#   docker build --build-arg BASE_IMAGE=python:3.11-slim-bookworm \
#                --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
#                -t nyc-taxi:cpu .
# Run:
#   docker run --rm --gpus all --env-file .env nyc-taxi:latest \
#       nyc-taxi-train training.max_epochs=30
# ==============================================================================

ARG PYTHON_VERSION=3.11
ARG BASE_IMAGE=nvidia/cuda:12.1.1-runtime-ubuntu22.04
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------------------------
# Stage 1: builder
# ------------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG TORCH_INDEX_URL

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Build-time system dependencies only (psycopg2 needs libpq + a C toolchain).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv "${VIRTUAL_ENV}" && pip install --upgrade pip setuptools wheel

WORKDIR /build

# --- Layer-caching boundary -------------------------------------------------
# Only the dependency manifest is copied first, so the (slow) dependency
# resolution layer is reused for every source-only change.
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p src/nyc_taxi && touch src/nyc_taxi/__init__.py \
    && pip install --extra-index-url "${TORCH_INDEX_URL}" .

# --- Application source -----------------------------------------------------
COPY src/ ./src/
RUN pip install --no-deps .

# ------------------------------------------------------------------------------
# Stage 2: runtime
# ------------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

ARG PYTHON_VERSION
ARG APP_USER=mluser
ARG APP_UID=10001

LABEL org.opencontainers.image.title="nyc-taxi-trip-analytics" \
      org.opencontainers.image.description="ETL + PyTorch training pipeline for NYC Yellow Taxi trips" \
      org.opencontainers.image.source="https://github.com/AndrewVynnytskyi/Nyc-taxi-trip" \
      org.opencontainers.image.licenses="MIT"

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    NYC_TAXI_DATA_DIR=/data \
    NYC_TAXI_ARTIFACT_DIR=/artifacts \
    WANDB_DIR=/artifacts/wandb \
    HF_HOME=/artifacts/.cache

# Runtime system dependencies only: the interpreter, libpq and TLS roots.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install --no-install-recommends -y \
        "python${PYTHON_VERSION}" \
        libpq5 \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf "/usr/bin/python${PYTHON_VERSION}" /usr/local/bin/python3 \
    && ln -sf "/usr/bin/python${PYTHON_VERSION}" /usr/local/bin/python

# Non-root execution.
RUN groupadd --gid "${APP_UID}" "${APP_USER}" \
    && useradd --uid "${APP_UID}" --gid "${APP_UID}" --create-home --shell /usr/sbin/nologin "${APP_USER}" \
    && mkdir -p /app "${NYC_TAXI_DATA_DIR}" "${NYC_TAXI_ARTIFACT_DIR}" \
    && chown -R "${APP_UID}:${APP_UID}" /app "${NYC_TAXI_DATA_DIR}" "${NYC_TAXI_ARTIFACT_DIR}"

COPY --from=builder --chown=${APP_UID}:${APP_UID} /opt/venv /opt/venv

WORKDIR /app
COPY --chown=${APP_UID}:${APP_UID} configs/ ./configs/
COPY --chown=${APP_UID}:${APP_UID} sql/ ./sql/

USER ${APP_USER}

VOLUME ["/data", "/artifacts"]

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import torch, nyc_taxi; print(torch.__version__)" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["nyc-taxi-train"]
