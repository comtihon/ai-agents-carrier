FROM python:3.12-slim

WORKDIR /app

# Install deps with layer caching: stub out the app package so setuptools resolves deps
COPY pyproject.toml .
RUN mkdir -p app && touch app/__init__.py \
    && pip install --no-cache-dir . \
    && rm -rf app

# Copy application source (workflows are mounted via ConfigMap at runtime)
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
