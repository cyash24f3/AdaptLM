FROM ghcr.io/astral-sh/uv:0.8.15 AS uv
FROM python:3.12.13-slim-bookworm AS fixture
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 ADAPTLM_PROFILE=fixture ADAPTLM_DATA_DIR=/app/datasets ADAPTLM_REPORT_DIR=/app/reports
COPY pyproject.toml uv.lock /app/
COPY src /app/src
RUN uv sync --frozen --no-dev --no-extra model --no-extra train
COPY datasets /app/datasets
COPY configs /app/configs
RUN mkdir -p /app/reports /app/models && useradd --uid 10001 --create-home app && chown -R app:app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/ready')"
CMD ["/app/.venv/bin/adaptlm", "serve", "--host", "0.0.0.0", "--port", "8000"]

# CPU inference: Linux Docker does not expose Apple's Metal GPU.
FROM fixture AS inference
USER root
RUN uv sync --frozen --no-dev --extra model
ENV ADAPTLM_PROFILE=local ADAPTLM_DEVICE=cpu ADAPTLM_DTYPE=float32 HF_HOME=/app/models/hf
USER app
