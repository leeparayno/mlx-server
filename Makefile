.PHONY: build run clean test benchmark help release test-quick

# Configuration
WORKSPACE := .swiftpm/xcode/package.xcworkspace
SCHEME := mlx-server
DESTINATION := 'platform=OS X'
DERIVED_DATA := $(shell find ~/Library/Developer/Xcode/DerivedData -name "mlx-server-*" -type d -maxdepth 1 2>/dev/null | head -1)
DEBUG_BIN := $(DERIVED_DATA)/Build/Products/Debug/mlx-server
RELEASE_BIN := $(DERIVED_DATA)/Build/Products/Release/mlx-server

# Default target
help:
	@echo "MLX Server Build Commands:"
	@echo ""
	@echo "  make build       - Build server with xcodebuild (required for Metal)"
	@echo "  make run         - Build and run the server"
	@echo "  make test-quick  - Build and run quick test with small model"
	@echo "  make release     - Build optimized release version"
	@echo "  make test        - Run all unit tests"
	@echo "  make benchmark   - Build and run benchmarks"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make install     - Install to /usr/local/bin"
	@echo ""
	@echo "NOTE: This project requires xcodebuild (not swift build) for Metal shader compilation."
	@echo ""

# Build in debug mode using xcodebuild (required for Metal shaders)
build:
	@echo "🔨 Building mlx-server (debug) with xcodebuild..."
	@echo "NOTE: Using xcodebuild to compile Metal shaders (swift build won't work)"
	xcodebuild build -workspace $(WORKSPACE) -scheme $(SCHEME) -destination $(DESTINATION)
	@echo "✅ Build complete"

# Build and run
run: build
	@echo "🚀 Running mlx-server..."
	$(DEBUG_BIN)

# Quick test with small model
test-quick: build
	@echo "🧪 Running quick test with Qwen2.5-0.5B..."
	$(DEBUG_BIN) --test --model mlx-community/Qwen2.5-0.5B-Instruct-4bit

# Build release version with optimizations
release:
	@echo "🔨 Building mlx-server (release, optimized)..."
	xcodebuild build -workspace $(WORKSPACE) -scheme $(SCHEME) -destination $(DESTINATION) -configuration Release
	@echo "✅ Release build complete"
	@echo "Binary location: $(RELEASE_BIN)"

# Run unit tests
test:
	@echo "🧪 Running unit tests..."
	xcodebuild test -workspace $(WORKSPACE) -scheme mlx-server-Package -destination $(DESTINATION)

# Build and run benchmarks
benchmark:
	@echo "🔨 Building benchmarks..."
	xcodebuild build -workspace $(WORKSPACE) -scheme mlx-benchmark -destination $(DESTINATION)
	@echo "📊 Running benchmarks..."
	@find ~/Library/Developer/Xcode/DerivedData -name "mlx-benchmark" -path "*/Debug/mlx-benchmark" -type f 2>/dev/null | head -1 | xargs -I {} {} --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --iterations 100

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	swift package clean
	rm -rf .build
	rm -rf ~/Library/Developer/Xcode/DerivedData/mlx-server-*
	@echo "✅ Clean complete"

# Install release binary to system
install: release
	@echo "📦 Installing mlx-server to /usr/local/bin..."
	cp $(RELEASE_BIN) /usr/local/bin/mlx-server
	@echo "✅ Done! Run with: mlx-server"

# Quick rebuild (clean + build)
rebuild: clean build

# Format code (requires swift-format)
format:
	@if command -v swift-format >/dev/null 2>&1; then \
		echo "✨ Formatting code..."; \
		find Sources Tests benchmarks -name "*.swift" -exec swift-format -i {} \; ; \
		echo "✅ Format complete"; \
	else \
		echo "❌ swift-format not installed. Install with: brew install swift-format"; \
	fi

# Show build configuration
info:
	@echo "Build Configuration:"
	@echo "  Workspace: $(WORKSPACE)"
	@echo "  Scheme: $(SCHEME)"
	@echo "  Derived Data: $(DERIVED_DATA)"
	@echo "  Debug Binary: $(DEBUG_BIN)"
	@echo "  Release Binary: $(RELEASE_BIN)"
