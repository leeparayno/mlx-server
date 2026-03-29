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
