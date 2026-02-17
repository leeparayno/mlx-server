#!/bin/bash
# load_test.sh - Test concurrent requests
# Phase 5.5 - Load testing for MLX Server

set -e

# Configuration
PORT=${PORT:-8080}
MODEL=${MODEL:-"mlx-community/Qwen2.5-0.5B-Instruct-4bit"}
NUM_REQUESTS=${NUM_REQUESTS:-100}
CONCURRENCY=${CONCURRENCY:-16}
MAX_TOKENS=${MAX_TOKENS:-50}

echo "=========================================="
echo "MLX Server Load Test"
echo "=========================================="
echo "Port: $PORT"
echo "Model: $MODEL"
echo "Requests: $NUM_REQUESTS"
echo "Concurrency: $CONCURRENCY"
echo "Max Tokens: $MAX_TOKENS"
echo "=========================================="

# Find the mlx-server executable
MLXSERVER=""
if [ -f "./mlx-server" ]; then
    MLXSERVER="./mlx-server"
elif [ -f "~/Library/Developer/Xcode/DerivedData/mlx-server-*/Build/Products/Debug/mlx-server" ]; then
    MLXSERVER=$(find ~/Library/Developer/Xcode/DerivedData -name "mlx-server" -type f -path "*/Build/Products/Debug/mlx-server" | head -n 1)
else
    echo "Error: mlx-server executable not found"
    echo "Please run 'make build' first"
    exit 1
fi

echo "Using executable: $MLXSERVER"

# Start server
echo ""
echo "Starting server..."
$MLXSERVER --model "$MODEL" --port $PORT &
SERVER_PID=$!

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "Error: Server did not start in time"
        exit 1
    fi
    sleep 1
done

# Run load test
echo ""
echo "Running $NUM_REQUESTS concurrent requests (concurrency: $CONCURRENCY)..."
START_TIME=$(date +%s)

seq 1 $NUM_REQUESTS | xargs -P $CONCURRENCY -I {} curl -s -X POST http://localhost:$PORT/v1/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"test\",\"prompt\":\"Hello world, this is request {}\",\"max_tokens\":$MAX_TOKENS,\"stream\":false}" \
    > /dev/null

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Calculate metrics
echo ""
echo "=========================================="
echo "Load Test Results"
echo "=========================================="
echo "Total Requests: $NUM_REQUESTS"
echo "Duration: ${DURATION}s"
echo "Throughput: $(echo "scale=2; $NUM_REQUESTS / $DURATION" | bc) req/s"
echo "Avg Latency: $(echo "scale=2; $DURATION / $NUM_REQUESTS * 1000" | bc) ms/req"
echo "=========================================="

# Check metrics endpoint
echo ""
echo "Server Metrics:"
curl -s http://localhost:$PORT/metrics | python3 -m json.tool 2>/dev/null || echo "Unable to parse metrics"

echo ""
echo "Load test complete!"
