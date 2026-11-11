FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv
COPY pyproject.toml ./
COPY assistant ./assistant
COPY alembic.ini ./
COPY migrations ./migrations
RUN pip install .
RUN useradd -m app && chown -R app /srv
USER app
EXPOSE 8000
# exec form so SIGTERM reaches uvicorn (graceful shutdown)
CMD ["uvicorn", "assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "20"]
