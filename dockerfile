# ---------- Stage 1 : Build dependencies ----------
FROM python:3.11-slim AS builder

WORKDIR /install

# Prevent cache and .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only production requirements
COPY requirements_prod.txt .

# Install dependencies to custom folder
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements_prod.txt


# ---------- Stage 2 : Runtime Image ----------
FROM python:3.11-slim

WORKDIR /app

# Copy installed python packages from builder stage
COPY --from=builder /install /usr/local

# Copy only required folders
COPY app/ ./app/
COPY models/ ./models/
COPY src/ ./src/
COPY uttils/ ./uttils/

# If model files exist
COPY models/ ./models/

# If configuration files needed
COPY app/config.py ./app/

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]