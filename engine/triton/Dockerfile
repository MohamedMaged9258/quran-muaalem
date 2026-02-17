FROM nvcr.io/nvidia/tritonserver:25.12-py3

# Installing uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# RUN apt-get update && apt-get install -y --no-install-recommends \
    # python3-venv \
    # && rm -rf /var/lib/apt/lists/*

# Install huggingface-cli
# RUN curl -LsSf https://hf.co/cli/install.sh | bash
# ENV PATH="/root/.local/bin:${PATH}"

# Create directories
RUN mkdir -p /models/muaalem/1 /opt/configs

# Copy scripts and config templates
COPY entry-level.sh /opt/entry-level.sh
COPY entrypoint.sh /opt/entrypoint.sh
COPY configs/* /opt/configs/
RUN chmod +x /opt/entry-level.sh /opt/entrypoint.sh

ENTRYPOINT ["/opt/entrypoint.sh"]
