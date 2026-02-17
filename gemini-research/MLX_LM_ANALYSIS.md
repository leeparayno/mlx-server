The mlx-lm repository is effectively a high-level Python interface that orchestrates the MLX framework.
To understand if you can get better performance by switching languages, it is critical to understand that Python is not doing the heavy lifting here.
1. What is mlx-lm doing in Python?
mlx-lm is "glue code." It handles the logic, but not the math.
• The Logic (Python): It loads weights, processes arguments, handles the text generation loop (append token \rightarrow predict next \rightarrow sample), and manages the tokenizer.
• The Math (C++ / Metal): When mlx-lm asks to multiply matrices (the 99% of the work), it hands that task off to the MLX core, which is written in efficient C++ and compiles kernels directly for the Apple Neural Engine and GPU via Metal.
The Bottleneck: On an M3 Ultra with 512GB of RAM running large models, your bottleneck is almost certainly Memory Bandwidth, not Python. The time it takes to move 100GB of weights from RAM to the GPU vastly outweighs the microseconds Python spends deciding which function to call next.
2. Can you rewrite it to be faster? (Swift, Rust, Mojo)
Yes, but the gains will likely be marginal (e.g., <5-10%) rather than revolutionary (2x).
A. Swift (Best "Native" Alternative)
Apple officially supports MLX Swift.
• Pros: You remove the Python Global Interpreter Lock (GIL) and interpreter overhead. This can slightly improve "latency" (time to first token) and stability, ensuring the GPU never waits for the CPU.
• Feasibility: High. The mlx-swift-examples repository already contains an implementation of LLM generation similar to mlx-lm.
• Verdict: If you want a native app experience without Python dependencies, use this. Don't expect a massive jump in tokens/s for large models.
B. Rust (mlx-rs)
There are community bindings like mlx-rs.
• Pros: Rust offers extreme safety and control over memory.
• Cons: The bindings are unofficial and less mature than the Python/Swift ones. You would be rewriting the "glue" logic in Rust, but still calling the same C++ MLX kernels underneath.
• Verdict: Fun for a project, but unlikely to beat the official Python/Swift implementations in raw speed.
C. Mojo
Mojo is designed to fix exactly this problem (writing systems code + glue code in one language).
• Status: Mojo support for Apple Silicon GPUs is still in active development (nightly builds).
• Verdict: Theoretically, Mojo could eventually allow you to write custom kernels that beat MLX's generic ones. Right now, it is not a drop-in replacement that will magically speed up your Llama-3 run.
3. How to actually get more performance (M3 Ultra Specifics)
Since you are on an M3 Ultra with 512GB RAM, your performance issues are likely configuration-related, not language-related.
1. The "Wired Memory" Limit (Crucial for 512GB Macs)
macOS often restricts how much RAM the GPU can "wire" (lock) for itself. If you load a 300GB model, the OS might swap or throttle it. You can force macOS to allow more wired memory for the GPU:

# Example: Allow ~400GB for GPU (Check your specific needs)
sudo sysctl iogpu.wired_limit_mb=409600


Note: This resets on reboot.
2. Quantization is King
On M3 Ultra, memory bandwidth is the limit. A 4-bit quantized model requires half the bandwidth of an 8-bit model, effectively doubling your speed. Ensure you are using mlx-lm with 4-bit quantized models (e.g., mlx-community/Llama-3-70B-4bit).
3. Batching
If you are serving multiple users, Python's overhead does matter. In that case, rewriting the server layer in C++ or Swift (or using MLX Server which optimizes this) allows for continuous batching, which significantly increases total throughput (though not single-user speed).
Summary:
Rewriting mlx-lm in Swift might make the generation loop 1-2ms faster per token, which is negligible if your model takes 100ms per token. Stick with Python for ease of use, or switch to Swift if you are building a standalone app.
... Apple Mac Studio M3 Ultra Review
This video is relevant because it specifically benchmarks the M3 Ultra Mac Studio with 512GB RAM using MLX and compares it against other hardware, giving you a realistic baseline for what performance "good" looks like on your specific machine.