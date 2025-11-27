# Dockerfile
FROM python:3.11-slim

# avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends git-lfs curl ca-certificates && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Set workdir and copy sources
WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Install dependencies (will pick CPU torch wheels via requirements.txt extra index)
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application files
COPY . /app

# Expose the FastAPI port
EXPOSE 8000

# Default command: run uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
