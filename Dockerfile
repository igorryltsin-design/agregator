# syntax=docker/dockerfile:1.6

# Stage 1: build front-end bundles
FROM --platform=$BUILDPLATFORM node:20-alpine AS web
WORKDIR /build

# Build main frontend
COPY frontend/package*.json frontend/
RUN cd frontend && npm ci
COPY frontend frontend
RUN cd frontend && npm run build

# Build AiWord frontend, exposed from /aiword
COPY AiWord/package*.json AiWord/
RUN cd AiWord && npm ci
COPY AiWord AiWord
RUN cd AiWord && npm run build

# Stage 2: runtime image with Python backend
FROM --platform=$TARGETPLATFORM python:3.11-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install system dependencies listed in docker/system-packages.txt
COPY docker/system-packages.txt /tmp/system-packages.txt
RUN apt-get update \
    && xargs -r -a /tmp/system-packages.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/* /tmp/system-packages.txt

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy application sources
COPY . ./

# Provide healthcheck helper
COPY docker/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh

# Bring in pre-built front-end assets from the web stage
COPY --from=web /build/frontend/dist ./frontend/dist
COPY --from=web /build/AiWord/dist ./AiWord/dist

EXPOSE 5050
CMD ["python", "app.py"]
