#!/bin/bash
REQ_ID=$1
PORT=$2
MAX_TOKENS=$3
RESULTS_DIR=$4

REQ_START=$(date +%s%3N)
curl -s -X POST http://localhost:$PORT/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"test\",\"prompt\":\"Load test request $REQ_ID\",\"max_tokens\":$MAX_TOKENS,\"stream\":false}" \
    > /dev/null 2>&1
REQ_END=$(date +%s%3N)
LATENCY=$((REQ_END - REQ_START))
echo "$REQ_ID,$LATENCY" >> "$RESULTS_DIR/latencies.csv"
