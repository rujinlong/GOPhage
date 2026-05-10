# GOPhage container: prodigal/diamond + Python 3.10 + GPU PyTorch + GOPhage CLI.
# Base image already provides CUDA 12.6 runtime libs.
FROM mambaorg/micromamba:cuda12.6.3-ubuntu24.04

LABEL author="Jinlong Ru" \
      tool="GOPhage" \
      description="Phage protein GO annotation via ESM2 + Transformer + DiamondBLASTp"

USER root
WORKDIR /opt/gophage

# Install system tools (prodigal, diamond, uv) via the bioconda/conda-forge channels.
COPY --chown=$MAMBA_USER:$MAMBA_USER gophage.yaml /tmp/gophage.yaml
RUN --mount=type=cache,target=/opt/conda/pkgs \
    micromamba install -y -n base -f /tmp/gophage.yaml && \
    micromamba clean --all --yes

# Make conda env binaries (prodigal, diamond, uv, python) visible to all subsequent layers.
ENV PATH="/opt/conda/bin:${PATH}"
ARG MAMBA_DOCKERFILE_ACTIVATE=1

# Install GPU PyTorch (CUDA 12.6 wheels) and the rest of the Python deps.
COPY pyproject.toml /opt/gophage/pyproject.toml
COPY src /opt/gophage/src
COPY README.md /opt/gophage/README.md

# uv pip install: stable across architectures, doesn't require lock file at build time.
# torch comes from the official cu126 index; everything else from PyPI.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      torch && \
    uv pip install --system /opt/gophage

# Default workdir for user-mounted data; bind-mount the GOPhage data bundle here.
WORKDIR /work

ENTRYPOINT ["gophage"]
CMD ["--help"]
