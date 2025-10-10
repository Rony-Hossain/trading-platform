FROM python:3.11-slim
WORKDIR /app
ENV POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1
RUN pip install --no-cache-dir poetry==1.8.3
COPY pyproject.toml /app/
RUN poetry install --no-root --only main
COPY app /app/app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
