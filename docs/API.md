# MLX Server API Documentation

**Version:** 1.0.0 (Phase 5)
**Base URL:** `http://localhost:8080` (default)
**Protocol:** HTTP/1.1
**Format:** JSON (except SSE streaming)

## Overview

MLX Server provides an OpenAI-compatible REST API for text generation and chat completions. The API supports both streaming (Server-Sent Events) and non-streaming modes.

## Authentication

**Phase 5:** No authentication required (local development)
**Phase 6:** API key authentication will be added

## Endpoints

### Table of Contents

- [Completions](#completions)
  - [Create Completion](#create-completion)
  - [Create Completion (Streaming)](#create-completion-streaming)
- [Chat](#chat)
  - [Create Chat Completion](#create-chat-completion)
  - [Create Chat Completion (Streaming)](#create-chat-completion-streaming)
- [Management](#management)
  - [Cancel Request](#cancel-request)
- [Observability](#observability)
  - [Health Check](#health-check)
  - [Readiness Check](#readiness-check)
  - [Metrics](#metrics)

---

## Completions

### Create Completion

Generate text completion for a given prompt.

**Endpoint:** `POST /v1/completions`

**Request Body:**

```json
{
  "model": "string",           // Model identifier (required)
  "prompt": "string",          // Text prompt (required)
  "max_tokens": 100,           // Maximum tokens to generate (optional, default: 100)
  "temperature": 0.7,          // Sampling temperature 0.0-2.0 (optional, default: 0.7)
  "stream": false              // Enable streaming (optional, default: false)
}
```

**Response:** `200 OK`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "object": "text_completion",
  "created": 1708214400,
  "model": "model-name",
  "choices": [
    {
      "text": "Generated text continuation...",
      "index": 0,
      "finish_reason": "stop"
    }
  ]
}
```

**Finish Reasons:**
- `stop` - Model hit natural stop
- `length` - Reached max_tokens limit

**Example (curl):**

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "prompt": "The quick brown fox",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Example (Python - OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    prompt="The quick brown fox",
    max_tokens=50,
    temperature=0.7
)

print(response.choices[0].text)
```

**Error Responses:**

```json
// 400 Bad Request - Empty prompt
{
  "error": true,
  "reason": "Prompt cannot be empty"
}

// 500 Internal Server Error - Scheduler not initialized
{
  "error": true,
  "reason": "Scheduler not initialized"
}
```

---

### Create Completion (Streaming)

Stream text completion tokens as they are generated.

**Endpoint:** `POST /v1/completions`

**Request Body:**

```json
{
  "model": "string",
  "prompt": "string",
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": true              // Set to true for streaming
}
```

**Response:** `200 OK`

**Content-Type:** `text/event-stream`

**Headers:**
```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive
X-Accel-Buffering: no
X-Request-ID: <uuid>
```

**Stream Format (SSE):**

```
data: {"id":"550e8400-e29b-41d4-a716-446655440000","object":"text_completion.chunk","created":1708214400,"model":"model-name","choices":[{"text":"The","index":0,"finish_reason":null}]}

data: {"id":"550e8400-e29b-41d4-a716-446655440000","object":"text_completion.chunk","created":1708214400,"model":"model-name","choices":[{"text":" quick","index":0,"finish_reason":null}]}

data: {"id":"550e8400-e29b-41d4-a716-446655440000","object":"text_completion.chunk","created":1708214400,"model":"model-name","choices":[{"text":" brown","index":0,"finish_reason":null}]}

data: {"id":"550e8400-e29b-41d4-a716-446655440000","object":"text_completion.chunk","created":1708214400,"model":"model-name","choices":[{"text":" fox","index":0,"finish_reason":"stop"}]}

data: [DONE]
```

**Example (curl):**

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "prompt": "The quick brown fox",
    "max_tokens": 50,
    "stream": true
  }'
```

**Example (Python - OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

stream = client.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    prompt="The quick brown fox",
    max_tokens=50,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].text:
        print(chunk.choices[0].text, end='', flush=True)
```

**Example (JavaScript - fetch):**

```javascript
const response = await fetch('http://localhost:8080/v1/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'mlx-community/Qwen2.5-0.5B-Instruct-4bit',
    prompt: 'The quick brown fox',
    max_tokens: 50,
    stream: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');

  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = line.slice(6);
      if (data === '[DONE]') break;

      const json = JSON.parse(data);
      console.log(json.choices[0].text);
    }
  }
}
```

---

## Chat

### Create Chat Completion

Generate chat completion for a conversation.

**Endpoint:** `POST /v1/chat/completions`

**Request Body:**

```json
{
  "model": "string",           // Model identifier (required)
  "messages": [                // Conversation messages (required)
    {
      "role": "system",        // Role: system, user, or assistant
      "content": "string"      // Message content
    },
    {
      "role": "user",
      "content": "string"
    }
  ],
  "max_tokens": 100,           // Maximum tokens to generate (optional)
  "temperature": 0.7,          // Sampling temperature (optional)
  "stream": false              // Enable streaming (optional)
}
```

**Supported Roles:**
- `system` - System instruction/context
- `user` - User message
- `assistant` - Assistant response (for multi-turn)

**Response:** `200 OK`

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "object": "chat.completion",
  "created": 1708214400,
  "model": "model-name",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "index": 0,
      "finish_reason": "stop"
    }
  ]
}
```

**Example (curl):**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
  }'
```

**Example (Python - OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
```

**Multi-Turn Conversation Example:**

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

# First turn
messages.append({"role": "user", "content": "What is 2+2?"})
response = client.chat.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    messages=messages,
    max_tokens=50
)
messages.append({"role": "assistant", "content": response.choices[0].message.content})

# Second turn
messages.append({"role": "user", "content": "What about 3+3?"})
response = client.chat.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    messages=messages,
    max_tokens=50
)
print(response.choices[0].message.content)
```

**Error Responses:**

```json
// 400 Bad Request - Empty messages
{
  "error": true,
  "reason": "Messages cannot be empty"
}
```

---

### Create Chat Completion (Streaming)

Stream chat completion tokens as they are generated.

**Endpoint:** `POST /v1/chat/completions`

**Request Body:**

```json
{
  "model": "string",
  "messages": [
    {"role": "system", "content": "string"},
    {"role": "user", "content": "string"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": true              // Set to true for streaming
}
```

**Response:** `200 OK`

**Content-Type:** `text/event-stream`

**Stream Format (SSE):**

```
data: {"id":"550e8400-e29b-41d4-a716-446655440000","object":"chat.completion.chunk","created":1708214400,"model":"model-name","choices":[{"delta":{"role":"assistant","content":"Hello"},"index":0,"finish_reason":null}]}

data: {"id":"550e8400-e29b-41d4-a716-446655440000","object":"chat.completion.chunk","created":1708214400,"model":"model-name","choices":[{"delta":{"role":"assistant","content":"!"},"index":0,"finish_reason":null}]}

data: {"id":"550e8400-e29b-41d4-a716-446655440000","object":"chat.completion.chunk","created":1708214400,"model":"model-name","choices":[{"delta":{"role":"assistant","content":""},"index":0,"finish_reason":"stop"}]}

data: [DONE]
```

**Example (curl):**

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "stream": true
  }'
```

**Example (Python - OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

stream = client.chat.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    messages=[{"role": "user", "content": "Tell me a story"}],
    max_tokens=200,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
```

---

## Management

### Cancel Request

Cancel an in-progress inference request.

**Endpoint:** `DELETE /v1/requests/:id`

**Parameters:**
- `id` (path parameter) - Request UUID to cancel

**Response:** `200 OK` (No content)

**Example (curl):**

```bash
curl -X DELETE http://localhost:8080/v1/requests/550e8400-e29b-41d4-a716-446655440000
```

**Error Responses:**

```json
// 400 Bad Request - Invalid UUID
{
  "error": true,
  "reason": "Invalid request ID format"
}

// 404 Not Found - Request not found
{
  "error": true,
  "reason": "Request not found"
}
```

**Notes:**
- Can cancel pending or active requests
- Cancellation is best-effort (may complete if nearly done)
- Cancelled requests appear in metrics as "cancelled"

---

## Observability

### Health Check

Check overall system health.

**Endpoint:** `GET /health`

**Response:** `200 OK`

```json
{
  "status": "healthy",         // "healthy" or "degraded"
  "model": "loaded",           // "loaded" or "not_loaded"
  "batcher": "running",        // "running" or "stopped"
  "timestamp": "2026-02-17T08:40:00Z"
}
```

**Status Values:**
- `healthy` - Model loaded and batcher running
- `degraded` - Model not loaded or batcher stopped

**Example (curl):**

```bash
curl http://localhost:8080/health
```

**Use Cases:**
- Load balancer health checks
- Monitoring/alerting
- Deployment readiness gates

---

### Readiness Check

Check if server is ready to accept requests.

**Endpoint:** `GET /ready`

**Response:**
- `200 OK` with body "Ready" - Server ready
- `503 Service Unavailable` with body "Not Ready" - Server not ready

**Example (curl):**

```bash
curl http://localhost:8080/ready
```

**Use Cases:**
- Kubernetes readiness probes
- Load balancer backend checks
- Pre-flight checks

---

### Metrics

Retrieve detailed server metrics.

**Endpoint:** `GET /metrics`

**Response:** `200 OK`

```json
{
  "requests": {
    "pending": 5,              // Requests waiting in queue
    "active": 3,               // Currently processing
    "completed": 1234,         // Successfully completed
    "failed": 12,              // Failed requests
    "cancelled": 3             // User-cancelled requests
  },
  "batcher": {
    "active_slots": 6,         // Slots with active requests
    "total_slots": 8,          // Total batch slots
    "utilization": 0.75,       // Slot utilization (0.0-1.0)
    "step_count": 5678         // Total batching steps
  },
  "gpu": {
    "average_utilization": 0.85,    // Average GPU usage
    "current_utilization": 0.92,    // Current GPU usage
    "sample_count": 100             // Number of samples
  }
}
```

**Example (curl):**

```bash
curl http://localhost:8080/metrics | python3 -m json.tool
```

**Example (Python):**

```python
import requests

response = requests.get('http://localhost:8080/metrics')
metrics = response.json()

print(f"Throughput: {metrics['batcher']['utilization']*100:.1f}%")
print(f"Queue depth: {metrics['requests']['pending']}")
print(f"GPU usage: {metrics['gpu']['current_utilization']*100:.1f}%")
```

**Use Cases:**
- Performance monitoring
- Capacity planning
- Debugging throughput issues
- Auto-scaling decisions

---

## Request Headers

### Standard Headers

All requests support standard HTTP headers:

```
Content-Type: application/json
Accept: application/json
User-Agent: <client-name>
```

### Custom Headers

**X-Request-ID**

- **Outgoing:** Optional - Provide your own request ID for tracing
- **Incoming:** Always present - Server adds UUID if not provided
- **Format:** UUID v4 string

**Example:**

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: my-custom-trace-id-123" \
  -d '{"model":"test","prompt":"Hello","max_tokens":50}'
```

The response will include the same X-Request-ID:

```
HTTP/1.1 200 OK
X-Request-ID: my-custom-trace-id-123
Content-Type: application/json
...
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | When |
|------|---------|------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request format or parameters |
| 404 | Not Found | Endpoint or resource not found |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Server not ready (model not loaded) |

### Error Response Format

```json
{
  "error": true,
  "reason": "Descriptive error message"
}
```

### Common Errors

**Empty Prompt:**
```json
{
  "error": true,
  "reason": "Prompt cannot be empty"
}
```

**Empty Messages:**
```json
{
  "error": true,
  "reason": "Messages cannot be empty"
}
```

**Scheduler Not Initialized:**
```json
{
  "error": true,
  "reason": "Scheduler not initialized"
}
```

**Invalid Request ID:**
```json
{
  "error": true,
  "reason": "Invalid request ID format"
}
```

---

## Rate Limiting

**Phase 5:** No rate limiting
**Phase 6:** Will add per-API-key rate limits

---

## OpenAI SDK Compatibility

MLX Server is fully compatible with the OpenAI Python SDK. Simply point the base URL to your MLX Server instance:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # Phase 5: no auth required
)

# Use exactly as you would with OpenAI
response = client.chat.completions.create(
    model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

**Supported Methods:**
- ✅ `client.completions.create()`
- ✅ `client.chat.completions.create()`
- ✅ Streaming with `stream=True`

**Not Yet Supported:**
- ❌ Embeddings
- ❌ Fine-tuning
- ❌ Files
- ❌ Images (DALL-E)
- ❌ Audio (Whisper, TTS)

---

## Performance Characteristics

### Throughput

- **Test Environment:** 68.58 req/s (50 concurrent)
- **Production:** Varies by model and hardware

### Latency

| Metric | Test Environment |
|--------|------------------|
| P50 | 15ms |
| P95 | 16ms |
| P99 | 16ms |
| TTFT (streaming) | <500ms |

### Concurrency

- **Tested:** 100+ concurrent requests
- **Batch Size:** Configurable (default: 8 slots)
- **Queue:** Unlimited (memory permitting)

---

## Testing Scripts

### Load Testing

Run load tests with configurable concurrency:

```bash
# Default: 100 requests, 16 concurrent
./scripts/load_test.sh

# Custom configuration
NUM_REQUESTS=200 CONCURRENCY=32 ./scripts/load_test.sh
```

### OpenAI SDK Compatibility

Test compatibility with actual OpenAI SDK:

```bash
# Requires: pip install openai
./scripts/test_openai_sdk.sh
```

---

## Changelog

### Phase 5 (February 2026)

**Added:**
- Initial API implementation
- `/v1/completions` endpoint
- `/v1/chat/completions` endpoint
- SSE streaming support
- `/health` and `/metrics` endpoints
- Request cancellation
- X-Request-ID tracking
- OpenAI SDK compatibility

---

## Support

For issues, questions, or feature requests:
- GitHub Issues: [MLX Server Issues](https://github.com/your-org/mlx-server/issues)
- Documentation: [MLX Server Docs](https://github.com/your-org/mlx-server/docs)

---

**Last Updated:** February 17, 2026
**API Version:** 1.0.0 (Phase 5)
