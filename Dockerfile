FROM python:3.12-slim

# Install PostgreSQL and pg_vector
RUN apt-get update && apt-get install -y \
    postgresql-client \
    postgresql-server-dev-all \
    postgresql \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pg_vector extension
RUN git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git /tmp/pgvector && \
    cd /tmp/pgvector && \
    make && \
    make install && \
    rm -rf /tmp/pgvector

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./

# Install uv and Python dependencies
RUN pip install uv && \
    uv sync

# Copy application code
COPY . .

# Expose PostgreSQL default port
EXPOSE 5432

# Default command
CMD ["postgres", "-D", "/var/lib/postgresql/data"]
