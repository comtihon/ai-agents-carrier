FROM python:3.12-slim

ARG HELM_VERSION=3.16.4

# Install system tools: uv for MCP servers, helm for K8s agent runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && pip install --no-cache-dir uv \
    && curl -fsSL "https://get.helm.sh/helm-v${HELM_VERSION}-linux-amd64.tar.gz" \
       | tar -xz -C /tmp \
    && mv /tmp/linux-amd64/helm /usr/local/bin/helm \
    && rm -rf /tmp/linux-amd64 \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade pip \
    && mkdir -p app && touch app/__init__.py \
    && pip install --no-cache-dir . \
    && rm -rf app

# Copy application source (graphs are mounted via ConfigMap at runtime)
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
