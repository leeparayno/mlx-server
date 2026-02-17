#!/bin/bash
# test_openai_sdk.sh - Test OpenAI SDK compatibility
# Phase 5.5 - OpenAI SDK compatibility testing

set -e

# Configuration
PORT=${PORT:-8080}
BASE_URL="http://localhost:$PORT/v1"

echo "=========================================="
echo "OpenAI SDK Compatibility Test"
echo "=========================================="
echo "Base URL: $BASE_URL"
echo "=========================================="

# Check if server is running
echo ""
echo "Checking server health..."
if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "Error: Server is not running at localhost:$PORT"
    echo "Please start the server first with: make run"
    exit 1
fi

echo "Server is healthy!"

# Run Python OpenAI SDK test
echo ""
echo "Testing OpenAI SDK compatibility..."
python3 <<EOF
import sys

try:
    import openai
except ImportError:
    print("Error: openai package not installed")
    print("Install with: pip install openai")
    sys.exit(1)

# Create client
client = openai.OpenAI(
    base_url="$BASE_URL",
    api_key="not-needed"  # API key not required for local testing
)

print("✓ OpenAI client created")

# Test 1: Completions endpoint
print("\nTest 1: Testing /v1/completions...")
try:
    response = client.completions.create(
        model="test",
        prompt="Hello, how are you?",
        max_tokens=20
    )
    print("✓ Completions request successful")
    print(f"  Response ID: {response.id}")
    print(f"  Generated text: {response.choices[0].text[:50]}...")
    print(f"  Finish reason: {response.choices[0].finish_reason}")
except Exception as e:
    print(f"✗ Completions test failed: {e}")
    sys.exit(1)

# Test 2: Chat completions endpoint
print("\nTest 2: Testing /v1/chat/completions...")
try:
    response = client.chat.completions.create(
        model="test",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ],
        max_tokens=20
    )
    print("✓ Chat completions request successful")
    print(f"  Response ID: {response.id}")
    print(f"  Generated text: {response.choices[0].message.content[:50]}...")
    print(f"  Finish reason: {response.choices[0].finish_reason}")
except Exception as e:
    print(f"✗ Chat completions test failed: {e}")
    sys.exit(1)

# Test 3: Streaming completions (non-blocking)
print("\nTest 3: Testing streaming completions...")
try:
    stream = client.completions.create(
        model="test",
        prompt="Count to 5:",
        max_tokens=30,
        stream=True
    )
    print("✓ Streaming completions initiated")

    tokens = []
    for chunk in stream:
        if chunk.choices[0].text:
            tokens.append(chunk.choices[0].text)
        if len(tokens) >= 5:  # Only collect first few tokens
            break

    print(f"  Received {len(tokens)} tokens: {''.join(tokens)[:50]}...")
    print("✓ Streaming works correctly")
except Exception as e:
    print(f"✗ Streaming test failed: {e}")
    sys.exit(1)

# Test 4: Streaming chat completions
print("\nTest 4: Testing streaming chat completions...")
try:
    stream = client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=20,
        stream=True
    )
    print("✓ Streaming chat completions initiated")

    tokens = []
    for chunk in stream:
        if chunk.choices[0].delta.content:
            tokens.append(chunk.choices[0].delta.content)
        if len(tokens) >= 5:  # Only collect first few tokens
            break

    print(f"  Received {len(tokens)} tokens: {''.join(tokens)[:50]}...")
    print("✓ Streaming chat works correctly")
except Exception as e:
    print(f"✗ Streaming chat test failed: {e}")
    sys.exit(1)

print("\n========================================")
print("✅ All OpenAI SDK compatibility tests passed!")
print("========================================")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "OpenAI SDK compatibility: PASSED"
    exit 0
else
    echo ""
    echo "OpenAI SDK compatibility: FAILED"
    exit 1
fi
