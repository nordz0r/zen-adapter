FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8100

WORKDIR /app

COPY pyproject.toml ./
COPY src/zen_adapter ./zen_adapter

RUN pip install --no-cache-dir -e .

RUN useradd --system --uid 10001 --no-create-home zen \
    && chown -R zen:zen /app

USER 10001

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8100')+'/healthz',timeout=3)"

CMD ["python", "-m", "uvicorn", "zen_adapter.app:app", "--host", "0.0.0.0", "--port", "8100"]