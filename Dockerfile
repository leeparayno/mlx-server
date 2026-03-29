# Multi-stage Docker build for MLX Server
# Stage 1: Build
FROM swift:6.2.3 as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY Package.swift Package.resolved ./

# Fetch dependencies (cached layer)
RUN swift package resolve

# Copy source code
COPY Sources ./Sources
COPY benchmarks ./benchmarks

# Build release binary
# Note: For Apple Silicon, this would need to run on macOS with xcodebuild
# For cross-platform deployment, consider using a build machine or CI/CD
RUN swift build -c release

# Stage 2: Runtime
FROM swift:6.2.3-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libxml2 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built binary from builder stage
COPY --from=builder /app/.build/release/mlx-server /usr/local/bin/

# Create cache directory for models
RUN mkdir -p /root/.cache/mlx-server/models

# Create non-root user (security best practice)
RUN useradd -m -u 1000 mlxserver && \
    chown -R mlxserver:mlxserver /root/.cache/mlx-server

# Switch to non-root user
USER mlxserver

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["mlx-server", "--port", "8080", "--model", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"]
