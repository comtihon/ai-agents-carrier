FROM python:3.14-slim

# Install uv (provides uvx) for running stdio MCP servers like mcp-atlassian
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "langgraph>=1.1.0,<2.0.0" \
        "langchain>=1.2.0,<2.0.0" \
        "copilotkit>=0.1.87,<1.0.0" \
        "starlette<0.47.0" \
        "uvicorn[standard]" \
        pydantic-settings \
        "pyjwt[crypto]" \
        motor \
        "pymongo>=4.12.0,<4.16.0" \
    && pip install --no-cache-dir \
        langchain-mcp-adapters \
        langchain-google-genai \
        "starlette<0.47.0" \
        "apscheduler>=3.10,<4.0" \
        "pyyaml>=6.0.3" \
    && mkdir -p app && touch app/__init__.py \
    && pip install --no-cache-dir --no-deps . \
    && rm -rf app

# Copy application source (graphs are mounted via ConfigMap at runtime)
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
