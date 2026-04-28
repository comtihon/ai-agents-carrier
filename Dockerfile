FROM python:3.12-slim

# Install uv (provides uvx) for running stdio MCP servers like mcp-atlassian
RUN pip install --no-cache-dir uv

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
