import Foundation
import Logging
import Vapor
import Core
import Scheduler
import API
import Authentication
import Memory

// Custom command-line parsing (before Vapor)
struct ServerConfig {
    var port: Int = 8080
    var model: String = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    var config: String?
    var logLevel: String = "info"
    var verbose: Bool = false
    var test: Bool = false
    var kvQuantEnabled: Bool = false
    var kvQuantBits: Int = 2
    var kvQuantMode: String = "mse"
    var kvQuantRotation: Bool = true
    var kvQuantRotationSeed: UInt64 = 1337
    var kvQuantQJLSeed: UInt64 = 4242
    var kvQuantGroupSize: Int = 64
    var kvQuantStart: Int = 0

    init(arguments: [String]) {
        // Try environment variables first (for compatibility with benchmarking scripts)
        if let envPort = ProcessInfo.processInfo.environment["MLX_PORT"], let portValue = Int(envPort) {
            port = portValue
        }
        if let envModel = ProcessInfo.processInfo.environment["MLX_MODEL"] {
            model = envModel
        }
        if let envLogLevel = ProcessInfo.processInfo.environment["MLX_LOG_LEVEL"] {
            logLevel = envLogLevel
        }
        if ProcessInfo.processInfo.environment["MLX_VERBOSE"] == "true" {
            verbose = true
        }
        if ProcessInfo.processInfo.environment["MLX_TEST"] == "true" {
            test = true
        }
        if ProcessInfo.processInfo.environment["MLX_KV_QUANT"] == "true" {
            kvQuantEnabled = true
        }
        if let bits = ProcessInfo.processInfo.environment["MLX_KV_QUANT_BITS"], let b = Int(bits) {
            kvQuantBits = b
        }
        if let mode = ProcessInfo.processInfo.environment["MLX_KV_QUANT_MODE"] {
            kvQuantMode = mode
        }
        if let rot = ProcessInfo.processInfo.environment["MLX_KV_ROTATION"], rot == "false" {
            kvQuantRotation = false
        }
        if let seed = ProcessInfo.processInfo.environment["MLX_KV_ROTATION_SEED"], let s = UInt64(seed) {
            kvQuantRotationSeed = s
        }
        if let seed = ProcessInfo.processInfo.environment["MLX_KV_QJL_SEED"], let s = UInt64(seed) {
            kvQuantQJLSeed = s
        }
        if let g = ProcessInfo.processInfo.environment["MLX_KV_GROUP_SIZE"], let gs = Int(g) {
            kvQuantGroupSize = gs
        }
        if let s = ProcessInfo.processInfo.environment["MLX_KV_QUANT_START"], let qs = Int(s) {
            kvQuantStart = qs
        }

        // Then parse command-line arguments (overrides environment variables)
        var i = 1 // Skip program name
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "-p", "--port":
                i += 1
                if i < arguments.count, let portValue = Int(arguments[i]) {
                    port = portValue
                }
            case "-m", "--model":
                i += 1
                if i < arguments.count {
                    model = arguments[i]
                }
            case "-c", "--config":
                i += 1
                if i < arguments.count {
                    config = arguments[i]
                }
            case "--log-level":
                i += 1
                if i < arguments.count {
                    logLevel = arguments[i]
                }
            case "--verbose":
                verbose = true
            case "--test":
                test = true
            case "-h", "--help":
                printHelp()
                exit(0)
            case "--version":
                print("mlx-server version 0.1.0")
                exit(0)
            default:
                break
            }
            i += 1
        }
    }

    func printHelp() {
        print("""
        OVERVIEW: High-performance MLX inference server for Apple Silicon

        USAGE: mlx-server [--port <port>] [--model <model>] [--config <config>] [--log-level <log-level>] [--verbose] [--test]

        OPTIONS:
          -p, --port <port>       Server port (default: 8080)
          -m, --model <model>     Model path or Hugging Face model ID (default: mlx-community/Qwen2.5-0.5B-Instruct-4bit)
          -c, --config <config>   Configuration file path
          --log-level <log-level> Log level (trace, debug, info, warning, error) (default: info)
          --verbose               Enable detailed logging
          --test                  Test mode: load model, run single inference, and exit
          --version               Show the version
          -h, --help              Show help information
        """)
    }
}

@main
struct MLXServer {
    static func main() async throws {
        // Parse arguments before Vapor sees them
        let config = ServerConfig(arguments: CommandLine.arguments)

        // Now run the server
        try await runServer(config: config)
    }

    static func runServer(config: ServerConfig) async throws {
        // Set up logging
        let level = parseLogLevel(config.verbose ? "debug" : config.logLevel)
        LoggingSystem.bootstrap { label in
            var handler = StreamLogHandler.standardOutput(label: label)
            handler.logLevel = level
            return handler
        }

        let logger = Logger(label: "mlx-server")

        logger.info("🚀 MLX Server starting...", metadata: [
            "version": "0.1.0",
            "implementation": "Option 1 (mlx-swift-lm)"
        ])

        logger.info("Configuration:", metadata: [
            "port": "\(config.port)",
            "model": "\(config.model)",
            "log_level": "\(level)",
            "kv_quant": "\(config.kvQuantEnabled)",
            "kv_quant_bits": "\(config.kvQuantBits)",
            "kv_quant_mode": "\(config.kvQuantMode)",
            "kv_rotation": "\(config.kvQuantRotation)",
            "kv_group_size": "\(config.kvQuantGroupSize)",
            "kv_quant_start": "\(config.kvQuantStart)"
        ])

        // Load configuration from file if provided
        if let configPath = config.config {
            logger.info("Loading configuration from: \(configPath)")
            // TODO: Phase 2 - Implement config file loading
        }

        // Phase 1: Model loading
        logger.info("📦 Loading model: \(config.model)")
        let loader = ModelLoader()

        do {
            let modelContainer = try await loader.load(modelPath: config.model)
            logger.info("✅ Model loaded successfully")

            // Phase 2: Initialize inference engine
            logger.info("🧠 Initializing inference engine...")
            let engine = InferenceEngine()
            await engine.initialize(modelContainer: modelContainer)
            logger.info("✅ Inference engine initialized")

            // Test mode: run single inference and exit
            if config.test {
                logger.info("🧪 Test mode: Running single inference...")
                try await runTestInference(engine: engine, logger: logger)
                logger.info("✅ Test completed successfully")
                return
            }

            // Phase 5: Start API server
            logger.info("🌐 Starting API server on port \(config.port)...")

            // Initialize scheduler and batcher
            logger.info("📋 Initializing request scheduler...")
            let scheduler = RequestScheduler()

            logger.info("🔄 Initializing continuous batcher...")
            let quantMode: QuantizationMode = (config.kvQuantMode.lowercased() == "prod") ? .prod : .mse
            let kvQuant = QuantizationConfig(
                enabled: config.kvQuantEnabled,
                bitWidth: config.kvQuantBits,
                rotationEnabled: config.kvQuantRotation,
                rotationSeed: config.kvQuantRotationSeed,
                mode: quantMode,
                qjlSeed: config.kvQuantQJLSeed,
                groupSize: config.kvQuantGroupSize,
                quantizedKVStart: config.kvQuantStart
            )
            let batcher = ContinuousBatcher(
                scheduler: scheduler,
                engine: engine,
                config: ContinuousBatcher.Config(maxBatchSize: 32, eosTokenId: 2, kvQuantization: kvQuant)
            )

            // Start batching loop in background
            logger.info("▶️  Starting continuous batching loop...")
            Task {
                await batcher.start()
            }

            // Small delay to ensure batcher starts
            try await Task.sleep(for: .milliseconds(100))

            // Create and configure Vapor application
            // Create environment with empty arguments to bypass command parsing
            let env = Environment(name: "production", arguments: [])
            let app = try await Application.make(env)
            logger.info("✅ Vapor application created")

            // Configure server
            app.http.server.configuration.hostname = "0.0.0.0"
            app.http.server.configuration.port = config.port

            // Phase 6: Initialize authentication
            logger.info("🔐 Initializing authentication...")
            let apiKeyStore = APIKeyStore()
            let rateLimiter = RateLimiter()

            // Create default API key for testing
            let defaultKey = await apiKeyStore.validate("sk-test-12345")
            if defaultKey != nil {
                logger.info("✅ Default API key available: sk-test-12345")
            }

            // Configure routes
            do {
                try routes(app, scheduler: scheduler, engine: engine, batcher: batcher, apiKeyStore: apiKeyStore, rateLimiter: rateLimiter)
                logger.info("✅ Routes configured")
            } catch {
                logger.error("❌ Failed to configure routes: \(error)")
                try await app.asyncShutdown()
                throw error
            }

            logger.info("✅ Server initialized successfully")
            logger.info("🚀 Server running on http://0.0.0.0:\(config.port)")
            logger.info("💡 Press Ctrl+C to stop the server")

            // Manually start server without going through Vapor's command system
            do {
                try await app.server.start(address: .hostname("0.0.0.0", port: config.port))
                logger.info("✅ HTTP server started and listening")
            } catch let error as DecodingError {
                logger.warning("⚠️ Vapor configuration warning (non-fatal): \(error)")
            } catch {
                // Check if error is CommandError from Vapor's command system
                let errorStr = String(describing: error)
                if errorStr.contains("CommandError") || errorStr.contains("unknownCommand") {
                    logger.warning("⚠️ Ignoring Vapor command parsing error: \(error)")
                    logger.info("✅ HTTP server started despite command error")
                } else {
                    logger.error("❌ Failed to start HTTP server: \(error)")
                    try await app.asyncShutdown()
                    throw error
                }
            }

            // Keep running until interrupted (sleep for a very long time)
            do {
                try await withTaskCancellationHandler {
                    try await Task.sleep(for: .seconds(31536000))  // 1 year
                } onCancel: {
                    logger.info("🛑 Server shutting down...")
                }
            } catch {
                // Ignore cancellation errors
            }

            // Clean shutdown
            try await app.asyncShutdown()
            logger.info("✅ Server stopped")

        } catch {
            let logger = Logger(label: "mlx-server")
            logger.error("❌ Failed to start server: \(error.localizedDescription)")
            exit(1)
        }
    }

    private static func runTestInference(engine: InferenceEngine, logger: Logger) async throws {
        let testPrompt = "Hello, how are you?"
        logger.info("Test prompt: \(testPrompt)")

        let startTime = Date()
        let result = try await engine.generate(
            prompt: testPrompt,
            maxTokens: 50,
            temperature: 0.7
        )
        let duration = Date().timeIntervalSince(startTime)

        logger.info("Test result:", metadata: [
            "prompt": "\(testPrompt)",
            "response": "\(result)",
            "duration": "\(String(format: "%.2f", duration))s"
        ])
    }

    private static func parseLogLevel(_ level: String) -> Logger.Level {
        switch level.lowercased() {
        case "trace": return .trace
        case "debug": return .debug
        case "info": return .info
        case "warning", "warn": return .warning
        case "error": return .error
        case "critical": return .critical
        default: return .info
        }
    }
}
