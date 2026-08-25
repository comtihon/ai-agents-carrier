# ai-agents-carrier — two images from one file.
#
#   slim (default)  the backend alone. Agent steps use the `docker` or `k8s`
#                   runtime, which spawn pi-cloud-agent as its own container.
#   full            slim plus a baked pi-cloud-agent, so the `local` runtime
#                   can run an agent as a child process of the backend.
#
#   docker build -t carrier:slim .
#   docker build -t carrier:full --target full .
#
# `full` trades isolation for not needing a container registry, a Helm chart or
# cluster permissions at run time. The agent then shares the backend's pod,
# filesystem and service-account token, so prefer `docker`/`k8s` wherever the
# agent runs model-authored code.

ARG PI_AGENT_IMAGE=ghcr.io/comtihon/pi-cloud-agent:latest

# ---------------------------------------------------------------------------
# base — everything both images share
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

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

# ---------------------------------------------------------------------------
# full — base + pi-cloud-agent, for LOCAL_AGENT_DIR
# ---------------------------------------------------------------------------
# Taken from the published agent image rather than rebuilt here, so the two
# always agree: one image is the single definition of what the agent is.
FROM ${PI_AGENT_IMAGE} AS agent

FROM base AS full

# node is dynamically linked against libstdc++, which python:3.12-slim omits.
# git/ripgrep/jq are what the agent's own tools shell out to.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 git ripgrep jq ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# The Node runtime and the agent itself. /opt/pi carries the baked pi
# extensions that seed-agent-home copies into HOME at start.
COPY --from=agent /usr/local/bin/node /usr/local/bin/node
COPY --from=agent /app /opt/pi-cloud-agent
COPY --from=agent /opt/pi /opt/pi
COPY --from=agent /usr/local/bin/git-credential-env /usr/local/bin/git-credential-env
COPY --from=agent /usr/local/bin/seed-agent-home /usr/local/bin/seed-agent-home

# HOME is where pi writes settings.json / mcp.json / sessions; it must exist
# and be writable by whoever the pod runs as, which is not necessarily uid 1000
# here — this image's main process is the backend, not the agent.
ENV LOCAL_AGENT_DIR=/opt/pi-cloud-agent \
    PI_BAKED_AGENT_DIR=/opt/pi/agent \
    PI_CODING_AGENT_DIR=/tmp/pi/agent \
    npm_config_cache=/tmp/.npm

RUN git config --system credential.helper env \
    && mkdir -p /workspace /tmp/pi/agent \
    && chmod -R a+rX /opt/pi /opt/pi-cloud-agent \
    && chmod 777 /workspace /tmp/pi /tmp/pi/agent

# ---------------------------------------------------------------------------
# slim — the default target, identical to base
# ---------------------------------------------------------------------------
# Last so that a plain `docker build .` still produces the backend-only image.
FROM base AS slim
