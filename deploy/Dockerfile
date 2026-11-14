# the command center as a container, built from the repository root
#   docker build -f deploy/Dockerfile -t neuse-command-center .
# it needs NEUSE_CC_TOKEN, and NEUSE_CC_DATABASE_URL for rds for mysql, see docs/DEPLOY.md
FROM python:3.12.16-slim

WORKDIR /app
COPY command_center/requirements.txt /app/command_center/requirements.txt
RUN pip install --no-cache-dir -r /app/command_center/requirements.txt

COPY command_center /app/command_center

# runs as an ordinary user, with a folder for the sqlite database when rds is not used
RUN useradd --create-home --uid 10001 neuse && mkdir -p /data && chown neuse /data
USER neuse

ENV NEUSE_CC_HOST=0.0.0.0 \
    NEUSE_CC_PORT=8000 \
    NEUSE_CC_DB=/data/command_center.db \
    PYTHONUNBUFFERED=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4)"

CMD ["python", "-m", "command_center.app"]
