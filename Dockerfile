# ── Trading Bot — Python application only ─────────────────────────────────────
#
# This image runs the Python trading bot (main.py / scripts/).
# It does NOT include IBKR Gateway — Gateway must run separately and be
# reachable at the host:port configured in your .env file.
#
# Typical setup on a VPS / cloud server:
#   1. Run IBKR Gateway in a Docker container or native install on the same host.
#   2. Build and run this image, passing your .env as a bind-mount or via
#      Docker secrets / environment variables.
#
# Build:
#   docker build -t trading-bot .
#
# Run (paper trading, Gateway on same host):
#   docker run -d \
#     --name trading-bot \
#     --network host \
#     -e IB_HOST=127.0.0.1 \
#     -e IB_PORT=7497 \
#     -e IB_CLIENT_ID=1 \
#     -v $(pwd)/logs:/app/logs \
#     -v $(pwd)/data:/app/data \
#     trading-bot
#
# Run (using a .env file):
#   docker run -d --name trading-bot --network host \
#     --env-file .env \
#     -v $(pwd)/logs:/app/logs \
#     -v $(pwd)/data:/app/data \
#     trading-bot

FROM python:3.11-slim

# Keeps Python output unbuffered so logs appear immediately in docker logs
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached layer unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Pre-create runtime directories so volume mounts don't need root
RUN mkdir -p logs data/cache

CMD ["python", "main.py"]
