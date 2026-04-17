FROM python:3.14-slim

# Install uv (provides uvx) for running stdio MCP servers like mcp-atlassian
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml .

# Install dependencies in two passes to work around a version conflict:
# copilotkit pins langchain<0.4 (langchain-core<1.0); langchain-mcp-adapters
# needs langchain-core>=1.0. Installing copilotkit's ecosystem first then
# upgrading langchain-core via mcp-adapters produces a working environment.
# copilotkit transitively pulls langchain/langchain-anthropic/langchain-openai/fastapi/httpx.
# langgraph<0.6.0 keeps langgraph-sdk in the <0.2.0 range required by copilotkit.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        copilotkit \
        "langgraph<0.6.0" \
        "starlette<0.47.0" \
        "uvicorn[standard]" \
        pydantic-settings \
        "pyjwt[crypto]" \
        motor \
    && pip install --no-cache-dir \
        langchain-mcp-adapters \
        langchain-google-genai \
        "starlette<0.47.0" \
        "apscheduler>=3.10,<4.0" \
        "pyyaml>=6.0.3" \
        pymongo \
    && mkdir -p app && touch app/__init__.py \
    && pip install --no-cache-dir --no-deps . \
    && rm -rf app

# Copy application source (graphs are mounted via ConfigMap at runtime)
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
