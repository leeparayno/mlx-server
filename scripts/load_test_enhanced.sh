#!/bin/bash
# load_test_enhanced.sh - Enhanced load testing with metrics
# Phase 4.4 Task 5 - Load testing framework

set -e

# Configuration
CONCURRENCY=${1:-16}
NUM_REQUESTS=${2:-100}
MAX_TOKENS=${3:-50}
MODEL=${MODEL:-"mlx-community/Qwen2.5-0.5B-Instruct-4bit"}
PORT=${PORT:-8080}
RESULTS_DIR="./test-results/load-test-c${CONCURRENCY}-$(date +%Y%m%d-%H%M%S)"

echo "=========================================="
echo "MLX Server Enhanced Load Test"
echo "=========================================="
echo "Concurrency: $CONCURRENCY"
echo "Requests: $NUM_REQUESTS"
echo "Max Tokens: $MAX_TOKENS"
echo "Model: $MODEL"
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

# Start server (using environment variables to avoid Vapor command parsing)
echo ""
echo "Starting server..."
MLX_MODEL="$MODEL" MLX_PORT=$PORT $MLXSERVER &> "$RESULTS_DIR/server.log" &
SERVER_PID=$!

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
    if [ -n "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "Error: Server did not start in time"
        tail -20 "$RESULTS_DIR/server.log"
        exit 1
    fi
    sleep 1
done

# Monitor memory in background
echo "timestamp,rss_kb" > "$RESULTS_DIR/memory.csv"
(
    while true; do
        if ps -p $SERVER_PID > /dev/null 2>&1; then
            RSS=$(ps -o rss= -p $SERVER_PID | tr -d ' ')
            echo "$(date +%s),$RSS" >> "$RESULTS_DIR/memory.csv"
        fi
        sleep 1
    done
) &
MONITOR_PID=$!

# Create request script
cat > "$RESULTS_DIR/run_request.sh" << 'EOF'
#!/bin/bash
REQ_ID=$1
PORT=$2
MAX_TOKENS=$3
RESULTS_DIR=$4

# Use Python for millisecond timing (date %N doesn't work on macOS)
REQ_START=$(python3 -c "import time; print(int(time.time() * 1000))")
curl -s -X POST http://localhost:$PORT/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"test\",\"prompt\":\"Load test request $REQ_ID\",\"max_tokens\":$MAX_TOKENS,\"stream\":false}" \
    > /dev/null 2>&1
REQ_END=$(python3 -c "import time; print(int(time.time() * 1000))")
LATENCY=$((REQ_END - REQ_START))
echo "$REQ_ID,$LATENCY" >> "$RESULTS_DIR/latencies.csv"
EOF

chmod +x "$RESULTS_DIR/run_request.sh"

# Run load test with detailed metrics
echo ""
echo "Running load test..."
echo "request_id,latency_ms" > "$RESULTS_DIR/latencies.csv"
START_TIME=$(date +%s)

seq 1 $NUM_REQUESTS | xargs -P $CONCURRENCY -I {} "$RESULTS_DIR/run_request.sh" {} $PORT $MAX_TOKENS "$RESULTS_DIR"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true
MONITOR_PID=""

# Get final metrics
METRICS=$(curl -s http://localhost:$PORT/metrics)
echo "$METRICS" > "$RESULTS_DIR/metrics.json"

# Calculate statistics using Python
python3 - "$RESULTS_DIR" "$CONCURRENCY" "$NUM_REQUESTS" "$MAX_TOKENS" "$DURATION" << 'PYTHON_SCRIPT'
import sys
import json
import csv
from pathlib import Path

results_dir = Path(sys.argv[1])
concurrency = int(sys.argv[2])
num_requests = int(sys.argv[3])
max_tokens = int(sys.argv[4])
duration = int(sys.argv[5])

# Read latencies
latencies = []
with open(results_dir / 'latencies.csv') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if len(row) == 2:
            latencies.append(float(row[1]))

latencies.sort()

# Calculate percentiles
def percentile(data, p):
    if not data:
        return 0
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[f]
    return data[f] + (k - f) * (data[c] - data[f])

p50 = percentile(latencies, 50)
p95 = percentile(latencies, 95)
p99 = percentile(latencies, 99)
avg = sum(latencies) / len(latencies) if latencies else 0
min_lat = min(latencies) if latencies else 0
max_lat = max(latencies) if latencies else 0

# Read memory stats
memory_samples = []
with open(results_dir / 'memory.csv') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if len(row) == 2:
            memory_samples.append(int(row[1]))

mem_min = min(memory_samples) if memory_samples else 0
mem_max = max(memory_samples) if memory_samples else 0
mem_avg = sum(memory_samples) / len(memory_samples) if memory_samples else 0

# Read server metrics
with open(results_dir / 'metrics.json') as f:
    metrics = json.load(f)

# Generate report
throughput = num_requests / duration if duration > 0 else 0

report = f"""# Load Test Report

**Date:** {Path(results_dir).name.split('-c')[0].replace('load-test-', '')}
**Concurrency:** {concurrency}
**Total Requests:** {num_requests}
**Max Tokens:** {max_tokens}
**Duration:** {duration}s

## Performance Results

- **Throughput:** {throughput:.2f} req/s
- **Average Latency:** {avg:.2f}ms

## Latency Distribution

- **Min:** {min_lat:.2f}ms
- **p50 (Median):** {p50:.2f}ms
- **p95:** {p95:.2f}ms
- **p99:** {p99:.2f}ms
- **Max:** {max_lat:.2f}ms

## Memory Usage

- **Min:** {mem_min/1024:.1f}MB
- **Max:** {mem_max/1024:.1f}MB
- **Avg:** {mem_avg/1024:.1f}MB
- **Growth:** {(mem_max-mem_min)/1024:.1f}MB

## Server Metrics

### Requests
- **Completed:** {metrics.get('requests', {}).get('completed', 0)}
- **Failed:** {metrics.get('requests', {}).get('failed', 0)}
- **Cancelled:** {metrics.get('requests', {}).get('cancelled', 0)}

### Batcher
- **Total Slots:** {metrics.get('batcher', {}).get('total_slots', 0)}
- **Step Count:** {metrics.get('batcher', {}).get('step_count', 0)}
- **Utilization:** {metrics.get('batcher', {}).get('utilization', 0):.1%}

### GPU
- **Average Utilization:** {metrics.get('gpu', {}).get('average_utilization', 0):.1%}
- **Sample Count:** {metrics.get('gpu', {}).get('sample_count', 0)}
"""

with open(results_dir / 'report.md', 'w') as f:
    f.write(report)

print(report)
PYTHON_SCRIPT

echo ""
echo "=========================================="
echo "Load test complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
