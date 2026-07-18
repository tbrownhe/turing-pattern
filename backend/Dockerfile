FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93 AS wheels

WORKDIR /build
COPY requirements.lock .
RUN pip wheel --no-cache-dir --require-hashes --wheel-dir /wheels \
    --requirement requirements.lock

FROM python:3.11-slim@sha256:db3ff2e1800a8581e2c48a27c3995339d47bdf046da21c7627accd3d51053a93

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 turing \
    && useradd --system --uid 10001 --gid turing --no-create-home turing

WORKDIR /app
COPY requirements.lock .
COPY --from=wheels /wheels /wheels
RUN pip install --no-cache-dir --no-index --require-hashes \
    --find-links=/wheels --requirement requirements.lock \
    && rm -rf /wheels

COPY --chown=turing:turing app ./app

USER turing
EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--ws-max-size", "4096"]
