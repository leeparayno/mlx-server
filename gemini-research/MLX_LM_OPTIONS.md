Referring to the possibility of “Write parts in other languages” — what’s realistically the best path on MLX?

If you want a “native” mlx-lm-like stack:

Option 1: Stay on MLX, move orchestration to Swift

MLX has a Swift API and MLX Swift exists specifically to run MLX natively in Swift.  
This is the cleanest way to remove Python overhead while keeping the same compute backend.

Expected payoff: usually modest for single-user chat; larger for multi-client serving and tighter latency control.

Option 2: Stay on MLX, move hot loops to C++

You can keep Python as the “front-end” but push:
	•	tokenization,
	•	sampling/logits processing,
	•	server loop,
into a native extension.

Expected payoff: similar to Swift; best when your Python layer is doing too much per token.

For these options what could be the performance expected from Swift vs C++