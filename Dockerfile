FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .

# Install dependencies in two passes to work around a version conflict:
# copilotkit pins langchain<0.4 (which requires langchain-core<1.0), while
# langchain-mcp-adapters requires langchain-core>=1.0. Installing copilotkit's
# ecosystem first and then upgrading langchain-core via mcp-adapters produces a
# working environment even though pip reports a post-install warning.
RUN pip install --no-cache-dir \
        "copilotkit>=0.1.39,<1.0.0" \
        "langchain>=0.3.28,<0.4.0" \
        "langchain-anthropic>=0.3.22,<0.4.0" \
        "langchain-openai>=0.3.35,<0.4.0" \
        "langgraph>=0.5.4,<0.6.0" \
        "uvicorn[standard]>=0.44.0,<1.0.0" \
        "pydantic-settings>=2.13.1,<3.0.0" \
        "pyjwt[crypto]>=2.12.1,<3.0.0" \
        "motor>=3.7.1,<4.0.0" \
    && pip install --no-cache-dir \
        "langchain-mcp-adapters>=0.2.2,<1.0.0" \
        "langchain-google-genai>=4.2.1,<5.0.0" \
    && mkdir -p app && touch app/__init__.py \
    && pip install --no-cache-dir --no-deps . \
    && rm -rf app

# Copy application source (graphs are mounted via ConfigMap at runtime)
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
