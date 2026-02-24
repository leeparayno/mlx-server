#!/bin/bash
# memory_test.sh - Memory leak testing for MLX Server
# Phase 4.4 Task 4 - Automated memory leak detection through 1000-request cycles

set -e

# Configuration
ITERATIONS=${ITERATIONS:-1000}
MODEL=${MODEL:-"mlx-community/Qwen2.5-0.5B-Instruct-4bit"}
PORT=${PORT:-8080}
MAX_TOKENS=${MAX_TOKENS:-10}
RESULTS_DIR="./test-results/memory-leak-$(date +%Y%m%d-%H%M%S)"

echo "=========================================="
echo "MLX Server Memory Leak Test"
echo "=========================================="
echo "Iterations: $ITERATIONS"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Results: $RESULTS_DIR"
echo "=========================================="

# Create results directory
mkdir -p "$RESULTS_DIR"

# Find the mlx-server executable
MLXSERVER=""
if [ -f "./mlx-server" ]; then
    MLXSERVER="./mlx-server"
else
    MLXSERVER=$(find ~/Library/Developer/Xcode/DerivedData -name "mlx-server" -type f -path "*/Build/Products/Debug/mlx-server" 2>/dev/null | head -n 1)
fi

if [ -z "$MLXSERVER" ]; then
    echo "Error: mlx-server executable not found"
    echo "Please run 'make build' first"
    exit 1
fi

echo "Using executable: $MLXSERVER"

# Start server
echo ""
echo "Starting server..."
# Use environment variables to avoid Vapor command parsing conflicts
MLX_MODEL="$MODEL" MLX_PORT=$PORT $MLXSERVER &
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
echo "Waiting for server to be ready (warmup 30s)..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "Error: Server did not start in time"
        exit 1
    fi
    sleep 1
done

# Additional warmup period
sleep 30

# Record baseline memory
BASELINE=$(ps -o rss= -p $SERVER_PID | tr -d ' ')
echo ""
echo "Baseline memory: ${BASELINE}KB ($(echo "scale=1; $BASELINE / 1024" | bc)MB)"
echo "Baseline memory: ${BASELINE}KB" > "$RESULTS_DIR/memory.log"
echo "" | tee -a "$RESULTS_DIR/memory.log"

# Run request cycles
echo ""
echo "Running $ITERATIONS requests (sampling every 100)..."
START_TIME=$(date +%s)

for i in $(seq 1 $ITERATIONS); do
    # Send request
    curl -s -X POST http://localhost:$PORT/v1/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"prompt\":\"Test $i\",\"max_tokens\":$MAX_TOKENS,\"stream\":false}" \
        > /dev/null

    # Sample memory every 100 requests
    if [ $((i % 100)) -eq 0 ]; then
        CURRENT=$(ps -o rss= -p $SERVER_PID | tr -d ' ')
        DELTA=$((CURRENT - BASELINE))
        DELTA_MB=$(echo "scale=1; $DELTA / 1024" | bc)
        echo "[$i/$ITERATIONS] Memory: ${CURRENT}KB ($(echo "scale=1; $CURRENT / 1024" | bc)MB, Δ${DELTA_MB}MB)" | tee -a "$RESULTS_DIR/memory.log"
    fi
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Final memory check
FINAL=$(ps -o rss= -p $SERVER_PID | tr -d ' ')
TOTAL_DELTA=$((FINAL - BASELINE))
TOTAL_DELTA_MB=$(echo "scale=1; $TOTAL_DELTA / 1024" | bc)
LEAK_RATE_KB=$(echo "scale=2; $TOTAL_DELTA / $ITERATIONS" | bc)
THROUGHPUT=$(echo "scale=2; $ITERATIONS / $DURATION" | bc)

echo "" | tee -a "$RESULTS_DIR/memory.log"
echo "=========================================="
echo "Memory Test Results"
echo "=========================================="
echo "Baseline: ${BASELINE}KB ($(echo "scale=1; $BASELINE / 1024" | bc)MB)" | tee -a "$RESULTS_DIR/memory.log"
echo "Final: ${FINAL}KB ($(echo "scale=1; $FINAL / 1024" | bc)MB)" | tee -a "$RESULTS_DIR/memory.log"
echo "Total Delta: ${TOTAL_DELTA}KB (${TOTAL_DELTA_MB}MB)" | tee -a "$RESULTS_DIR/memory.log"
echo "Leak Rate: ${LEAK_RATE_KB}KB/request" | tee -a "$RESULTS_DIR/memory.log"
echo "Duration: ${DURATION}s" | tee -a "$RESULTS_DIR/memory.log"
echo "Throughput: ${THROUGHPUT} req/s" | tee -a "$RESULTS_DIR/memory.log"
echo "=========================================="

# Verdict
echo "" | tee -a "$RESULTS_DIR/memory.log"
if [ "$TOTAL_DELTA" -lt 10240 ]; then  # <10MB total growth
    echo "✅ PASS: No memory leak detected (growth: ${TOTAL_DELTA_MB}MB)" | tee -a "$RESULTS_DIR/memory.log"
    EXIT_CODE=0
else
    echo "❌ FAIL: Memory leak detected (growth: ${TOTAL_DELTA_MB}MB)" | tee -a "$RESULTS_DIR/memory.log"
    EXIT_CODE=1
fi

echo ""
echo "Full results saved to: $RESULTS_DIR/memory.log"

exit $EXIT_CODE
