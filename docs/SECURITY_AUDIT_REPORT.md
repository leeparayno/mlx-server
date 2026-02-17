# MLX-Server Security Audit Report
**Local LLM Inference Engine - Apple Silicon M3 Ultra**

**Audit Date:** February 15, 2026  
**Auditor:** Security-Sentinel Agent  
**Project Phase:** Pre-implementation Research  
**Threat Model:** Local tool loading untrusted model files

---

## Executive Summary

Comprehensive security audit covering seven critical attack surfaces. Despite being local-only without network exposure, this system processes untrusted model files and user inputs, creating significant security risks.

**Overall Risk Rating: HIGH**

Primary threat vectors:
1. Untrusted model file loading (supply chain attacks)
2. User prompt injection attacks  
3. Memory safety in low-level operations
4. Resource exhaustion (DoS)

---

## Critical Findings

### 1. Model File Validation - CRITICAL RISK

**Threat:** Malicious/tampered model files causing system compromise

**Attack Vectors:**
- Model poisoning with adversarial weights
- Backdoored models with trigger-based behavior
- Tensor bomb attacks (memory exhaustion)
- Supply chain attacks from compromised sources
- Deserialization exploits

**Framework Risks:**

**Safetensors (MLX/Candle/Rust):** Designed for security but requires validation
- Header parsing buffer overflows
- Size field integer overflows  
- Metadata injection attacks
- Known issue: versions < 0.4.1 had header vulnerabilities (FIXED in 0.4.1+)

**GGUF (llama.cpp):** Binary format with parsing risks
- Complex header parsing
- Magic number spoofing potential
- Quantization metadata manipulation

**PyTorch .pt/.pth:** EXTREMELY DANGEROUS
- Uses Python serialization allowing arbitrary code execution
- NEVER load .pt files from untrusted sources
- Always convert to safetensors format

**MANDATORY Security Controls:**

1. **Checksum Verification**


```python
import hashlib

def verify_model_checksum(model_path: str, expected_sha256: str) -> bool:
    """Verify model integrity before loading."""
    sha256_hash = hashlib.sha256()
    
    with open(model_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)
    
    computed = sha256_hash.hexdigest()
    if computed != expected_sha256:
        raise SecurityError(f"Checksum mismatch: {computed} != {expected_sha256}")
    
    return True

# Store checksums in separate signed manifest
TRUSTED_MODELS = {
    "llama-3.1-8b-4bit": "a1b2c3d4e5f6...",
    "mistral-7b-q4": "f1e2d3c4b5a6...",
}
```

2. **Safetensors Header Validation**

```python
import struct
import json

def validate_safetensors_header(file_path: str) -> dict:
    """Validate safetensors header to prevent attacks."""
    with open(file_path, "rb") as f:
        # Read header length
        header_bytes = f.read(8)
        if len(header_bytes) != 8:
            raise SecurityError("Truncated file")
        
        header_length = struct.unpack("<Q", header_bytes)[0]
        
        # Sanity check: max 100MB header
        if header_length > 100 * 1024 * 1024:
            raise SecurityError(f"Header too large: {header_length}")
        
        # Parse JSON header
        header_json = f.read(header_length)
        if len(header_json) != header_length:
            raise SecurityError("Header length mismatch")
        
        header = json.loads(header_json)
        
        # Validate structure
        if not isinstance(header, dict):
            raise SecurityError("Invalid header format")
        
        # Check for injection attempts
        for key in header.keys():
            if key.startswith("__") and key != "__metadata__":
                raise SecurityError(f"Suspicious key: {key}")
        
        return header
```

3. **Multi-Layer Defense Pipeline**

```python
def secure_model_load(model_id: str, model_path: str, expected_checksum: str):
    """Defense-in-depth model loading."""
    
    # Layer 1: Source validation
    validate_model_source(model_id)
    
    # Layer 2: Path traversal prevention
    validate_local_model_path(model_path)
    
    # Layer 3: Checksum verification
    verify_model_checksum(model_path, expected_checksum)
    
    # Layer 4: Format validation
    if model_path.endswith(".safetensors"):
        header = validate_safetensors_header(model_path)
    elif model_path.endswith(".gguf"):
        header = validate_gguf_header(model_path)
    else:
        raise SecurityError(f"Unsupported format")
    
    # Layer 5: Load model
    model, tokenizer = load(model_path)
    
    # Layer 6: Weight sanity checks
    detect_suspicious_weights(model.parameters())
    
    return model, tokenizer
```

**Recommendations:**
✓ Implement checksum verification for ALL model loads
✓ Store checksums in separate cryptographically signed manifest
✓ Validate file format headers before parsing
✓ Whitelist trusted model sources only
✓ Convert PyTorch models to safetensors format
✓ Implement heuristic malicious weight detection

---

### 2. Memory Safety - HIGH RISK

**Threat:** Memory corruption leading to code execution or crashes

**Attack Vectors:**
- Buffer overflows in quantized operations
- Use-after-free in Swift ARC
- Double-free in Mojo manual memory management
- Integer overflows in size calculations
- Heap corruption from incorrect allocations

**Language-Specific Mitigations:**

**MLX (C++/Python/Swift):**

```swift
// Swift: Avoid retain cycles
class InferenceEngine {
    private weak var delegate: InferenceDelegate?  // weak reference
    
    func loadAsync(completion: @escaping (MLXModel) -> Void) {
        DispatchQueue.global().async { [weak self] in
            guard let self = self else { return }
            completion(self.loadModel())
        }
    }
}
```

```python
# Python: Explicit cleanup
import mlx.core as mx

def process_batch(prompts):
    model = load_model()
    results = []
    
    for prompt in prompts:
        output = model.generate(prompt)
        results.append(output)
        mx.synchronize(output)  # Release intermediate tensors
    
    del model
    return results
```

**Rust (Candle/mistral.rs):**

```rust
use candle_core::{Tensor, Error};

// Safe operations with Result types
fn safe_matrix_multiply(a: &Tensor, b: &Tensor) -> Result<Tensor, Error> {
    a.matmul(b)  // Returns Result, not panic
}

// Minimize unsafe blocks
unsafe fn zero_copy_view(data: &[f32]) -> Tensor {
    // SAFETY: Caller must guarantee:
    // 1. data valid for 'static lifetime
    // 2. data not modified during tensor lifetime
    // 3. data properly aligned
    Tensor::from_slice_unsafe(data)
}
```

**Mojo (Manual Memory):**

```mojo
# RAII pattern for automatic cleanup
struct SafeTensor[dtype: DType]:
    var data: Pointer[Scalar[dtype]]
    var size: Int
    
    fn __init__(inout self, size: Int) raises:
        if size <= 0 or size > 1_000_000_000:
            raise Error("Invalid size")
        self.size = size
        self.data = Pointer[Scalar[dtype]].alloc(size)
    
    fn __del__(owned self):
        self.data.free()  # Automatic cleanup
```

**Metal Shader Safety:**

```metal
// Bounds-checked kernel
kernel void safe_matmul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    
    // SECURITY: Bounds checking
    if (gid.x >= N || gid.y >= M) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        uint a_idx = gid.y * K + k;
        uint b_idx = k * N + gid.x;
        if (a_idx < M * K && b_idx < K * N) {
            sum += a[a_idx] * b[b_idx];
        }
    }
    
    uint out_idx = gid.y * N + gid.x;
    if (out_idx < M * N) {
        output[out_idx] = sum;
    }
}
```

**Recommendations:**
✓ Minimize unsafe blocks in Rust, document safety invariants
✓ Use weak references in Swift to prevent cycles
✓ Implement RAII patterns in Mojo
✓ Add bounds checking to all Metal kernels
✓ Use AddressSanitizer during development
✓ Validate all buffer sizes before GPU operations

Development tools:
```bash
# Rust with AddressSanitizer
RUSTFLAGS="-Z sanitizer=address" cargo build

# Swift with sanitizer
swift build -Xswiftc -sanitize=address

# Metal validation
export MTL_DEBUG_LAYER=1
export MTL_SHADER_VALIDATION=1
```

---

### 3. Input Validation - HIGH RISK

**Threat:** Malicious prompts causing harmful outputs or DoS

**Attack Vectors:**
- Prompt injection attacks
- Token overflow (memory exhaustion)
- Unicode exploits
- Control character injection
- Encoding attacks

**Prompt Sanitization:**

```python
import re
import unicodedata

class PromptSanitizer:
    MAX_PROMPT_LENGTH = 32768
    MAX_TOKENS = 4096
    
    SUSPICIOUS_PATTERNS = [
        r"ignore (all )?previous instructions",
        r"system:?.*\n",
        r"<\|im_start\|>",
        r"\[INST\]",
    ]
    
    def sanitize(self, prompt: str) -> str:
        """Apply all sanitization steps."""
        prompt = self._validate_encoding(prompt)
        prompt = self._validate_length(prompt)
        prompt = self._remove_control_chars(prompt)
        prompt = self._normalize_unicode(prompt)
        self._check_injection_patterns(prompt)
        return prompt
    
    def _remove_control_chars(self, prompt: str) -> str:
        """Remove ANSI escape sequences and control chars."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        prompt = ansi_escape.sub('', prompt)
        
        allowed = {'\n', '\t', '\r'}
        return ''.join(
            c for c in prompt
            if c in allowed or not unicodedata.category(c).startswith('C')
        )
    
    def _normalize_unicode(self, prompt: str) -> str:
        """Normalize to prevent homograph attacks."""
        return unicodedata.normalize('NFC', prompt)
```

**Token Limits:**

```python
class TokenLimiter:
    def __init__(self, max_prompt_tokens: int = 4096, max_gen_tokens: int = 2048):
        self.max_prompt_tokens = max_prompt_tokens
        self.max_gen_tokens = max_gen_tokens
    
    def truncate_prompt(self, tokens: list, strategy: str = "tail") -> list:
        if len(tokens) <= self.max_prompt_tokens:
            return tokens
        
        if strategy == "tail":
            return tokens[-self.max_prompt_tokens:]
        elif strategy == "head":
            return tokens[:self.max_prompt_tokens]
        elif strategy == "middle":
            half = self.max_prompt_tokens // 2
            return tokens[:half] + tokens[-half:]
```

**Recommendations:**
✓ Sanitize all user prompts
✓ Enforce maximum prompt length (chars and tokens)
✓ Remove control characters and ANSI escapes
✓ Normalize Unicode
✓ Log suspicious injection patterns
✓ Implement token count limits

---

### 4. File I/O Security - HIGH RISK

**Threat:** Unauthorized file system access

**Attack Vectors:**
- Path traversal (../../etc/passwd)
- Symlink attacks
- TOCTOU races
- Arbitrary file write
- Directory traversal

**Secure File Operations:**

```python
from pathlib import Path

class SecureFileLoader:
    def __init__(self):
        self.allowed_dirs = [
            Path.home() / ".cache" / "huggingface",
            Path.home() / ".cache" / "mlx",
            Path("/opt/mlx-models"),
        ]
    
    def validate_path(self, file_path: str) -> Path:
        """Validate path against security rules."""
        resolved = Path(file_path).resolve(strict=False)
        
        # Verify within allowed directories
        for allowed_dir in self.allowed_dirs:
            try:
                resolved.relative_to(allowed_dir)
                return resolved
            except ValueError:
                continue
        
        raise SecurityError(f"Path outside allowed dirs: {resolved}")
    
    def safe_open(self, file_path: str, mode: str = 'r'):
        validated = self.validate_path(file_path)
        if 'r' in mode and not validated.exists():
            raise FileNotFoundError(f"Not found: {validated}")
        return open(validated, mode)
```

**Atomic File Writing:**

```python
class AtomicFileWriter:
    def __init__(self, target_path: Path):
        self.target_path = target_path
        self.temp_path = target_path.with_suffix('.tmp')
    
    def __enter__(self):
        self.file = open(self.temp_path, 'wb')
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        if exc_type is None:
            self.temp_path.replace(self.target_path)
        else:
            self.temp_path.unlink(missing_ok=True)
```

**Recommendations:**
✓ Validate all paths before operations
✓ Restrict to allowed directories
✓ Check for symlink attacks
✓ Use atomic file writes
✓ Never construct paths from untrusted input

---

### 5. Resource Limits - MEDIUM RISK

**Threat:** Resource exhaustion causing DoS

**Attack Vectors:**
- Memory exhaustion (too many models)
- Infinite generation loops
- KV cache bombs
- Concurrent request floods

**Resource Management:**

```python
import mlx.core as mx
import signal
from contextlib import contextmanager

class ResourceManager:
    def __init__(self, max_memory_gb: float = 256.0):
        self.max_memory_gb = max_memory_gb
        self.active_requests = 0
        self.max_concurrent = 10
    
    def check_memory_available(self, required_gb: float) -> bool:
        active_gb = mx.metal.get_active_memory() / (1024**3)
        return (self.max_memory_gb - active_gb) >= required_gb
    
    def can_accept_request(self) -> bool:
        return self.active_requests < self.max_concurrent

@contextmanager
def timeout(seconds: int):
    """Operation timeout context manager."""
    def handler(signum, frame):
        raise TimeoutError(f"Exceeded {seconds}s")
    
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

# Usage
with timeout(300):  # 5 minute limit
    result = model.generate(prompt)
```

**Rate Limiting:**

```python
from collections import deque
import time

class RateLimiter:
    def __init__(self, max_requests: int = 100, window_sec: int = 60):
        self.max_requests = max_requests
        self.window = window_sec
        self.requests = deque()
    
    def acquire(self) -> bool:
        now = time.time()
        while self.requests and self.requests[0] < now - self.window:
            self.requests.popleft()
        
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(now)
        return True
```

**Recommendations:**
✓ Set maximum memory limits
✓ Enforce KV cache size limits
✓ Implement generation timeouts
✓ Limit concurrent requests
✓ Implement rate limiting
✓ Monitor memory continuously

---

### 6. Dependency Security - HIGH RISK

**Threat:** Vulnerable or compromised dependencies

**Attack Vectors:**
- Known CVEs
- Supply chain attacks
- Dependency confusion
- Typosquatting

**Critical Dependencies:**

Python (MLX-LM):
```toml
[project]
dependencies = [
    "mlx>=0.21.0",
    "huggingface_hub",
    "safetensors>=0.4.1",  # CRITICAL: Use 0.4.1+
    "transformers>=4.36.0",
    "pyyaml>=6.0",
]
```

Known vulnerabilities:
- safetensors < 0.4.1: Header parsing (FIXED)
- pyyaml < 6.0: Deserialization issues
- transformers < 4.36.0: Injection risks

Rust (Candle):
```toml
[dependencies]
candle-core = "=0.8.0"  # Pin exact versions
safetensors = "=0.4.5"
```

**Dependency Auditing:**

```bash
# Python
pip install pip-audit safety
pip-audit
safety check --full-report

# Rust
cargo install cargo-audit
cargo audit
```

**Secure Installation:**

```python
# requirements.txt with hashes
mlx==0.21.0 \
    --hash=sha256:abc123...
safetensors==0.4.5 \
    --hash=sha256:def456...

# Install with verification
pip install --require-hashes -r requirements.txt
```

**Recommendations:**
✓ Pin all dependencies with hashes
✓ Run pip-audit/cargo-audit in CI
✓ Only use safetensors (never PyTorch serialization)
✓ Audit dependencies before adding
✓ Enable automated updates
✓ Use Software Composition Analysis

---

### 7. Metal Shader Security - MEDIUM RISK

**Threat:** GPU kernel vulnerabilities

**Attack Vectors:**
- Buffer overflows
- Integer overflows
- Race conditions
- Uninitialized memory

**Safe Kernel Patterns:**

```metal
kernel void safe_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    // Bounds checking
    if (gid >= size) return;
    
    // Safe operations
    output[gid] = input[gid] * 2.0f;
}
```

```swift
// Swift validation wrapper
func safeDispatch(bufferA: MTLBuffer, bufferB: MTLBuffer, size: Int) throws {
    let requiredSize = size * MemoryLayout<Float>.stride
    guard bufferA.length >= requiredSize else {
        throw MetalError.bufferTooSmall
    }
    // ... dispatch kernel
}
```

**Recommendations:**
✓ Validate buffer sizes before dispatch
✓ Implement bounds checking in kernels
✓ Validate thread group dimensions
✓ Enable Metal validation in development

```bash
export MTL_DEBUG_LAYER=1
export MTL_SHADER_VALIDATION=1
```

---

## Security Checklist

### Pre-Deployment Audit

**Model Validation**
- [ ] Checksum verification implemented
- [ ] Safetensors header validation
- [ ] Model source whitelist configured
- [ ] Never using PyTorch serialization format

**Memory Safety**
- [ ] Rust unsafe blocks minimized
- [ ] Swift retain cycles prevented
- [ ] Mojo RAII patterns used
- [ ] Metal kernels bounds-checked
- [ ] Sanitizers enabled

**Input Validation**
- [ ] Prompt sanitization active
- [ ] Token limits enforced
- [ ] Control char removal
- [ ] Unicode normalization

**File I/O Security**
- [ ] Path traversal prevention
- [ ] Symlink detection
- [ ] Atomic file writes
- [ ] Allowed directories configured

**Resource Limits**
- [ ] Memory limits configured
- [ ] Generation timeouts enforced
- [ ] Concurrent limits set
- [ ] Rate limiting active

**Dependency Security**
- [ ] Versions pinned with hashes
- [ ] Audit tools in CI
- [ ] Safetensors-only policy
- [ ] SCA scanning enabled

**Metal Security**
- [ ] Buffer validation
- [ ] Kernel bounds checks
- [ ] GPU validation enabled

**Monitoring**
- [ ] Security logging configured
- [ ] Alerting thresholds set
- [ ] Incident response documented

---

## Priority Actions

**IMMEDIATE (Before coding):**
1. Implement model checksum verification
2. Configure allowed directories
3. Set up dependency pinning
4. Enable sanitizers

**HIGH PRIORITY (During development):**
1. Implement prompt sanitization
2. Add resource limits/timeouts
3. Metal kernel bounds checking
4. Security logging

**ONGOING:**
1. Automated security scans in CI
2. Monitor security logs
3. Dependency updates
4. Regular security reviews

---

## Residual Risks

Some risks will remain:
- Model poisoning detection is heuristic
- Zero-day vulnerabilities possible
- Hardware vulnerabilities in Apple Silicon
- Novel attack vectors

Document and accept these with stakeholders.

---

**Report Version:** 1.0  
**Audit Date:** February 15, 2026  
**Next Review:** Before production deployment  
**Classification:** Internal Use Only
