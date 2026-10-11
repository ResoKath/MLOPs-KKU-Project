FROM python:3.13.5-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PREFECT_HOME=/app/artifacts/prefect PREFECT_SERVER_ANALYTICS_ENABLED=false PREFECT_LOGGING_LEVEL=INFO
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY food_delivery /app/food_delivery
COPY input /app/input
COPY pyproject.toml /app/pyproject.toml
RUN mkdir -p /app/artifacts
EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=180s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"
CMD ["python", "-m", "food_delivery.launch"]
