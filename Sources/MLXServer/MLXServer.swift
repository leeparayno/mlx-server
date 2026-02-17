import Foundation
import ArgumentParser
import Logging
import Vapor
import Core
import Scheduler
import API

@main
struct MLXServerCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "mlx-server",
        abstract: "High-performance MLX inference server for Apple Silicon",
        version: "0.1.0"
    )

    @ArgumentParser.Option(name: .shortAndLong, help: "Server port")
    var port: Int = 8080

    @ArgumentParser.Option(name: .shortAndLong, help: "Model path or Hugging Face model ID")
    var model: String = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

    @ArgumentParser.Option(name: .shortAndLong, help: "Configuration file path")
    var config: String?

    @ArgumentParser.Option(name: .long, help: "Log level (trace, debug, info, warning, error)")
    var logLevel: String = "info"

    @ArgumentParser.Flag(name: .long, help: "Enable detailed logging")
    var verbose: Bool = false

    @ArgumentParser.Flag(name: .long, help: "Test mode: load model, run single inference, and exit")
    var test: Bool = false

    func run() async throws {
        // Set up logging
        let level = parseLogLevel(verbose ? "debug" : logLevel)
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
            "port": "\(port)",
            "model": "\(model)",
            "log_level": "\(level)"
        ])

        // Load configuration from file if provided
        if let configPath = config {
            logger.info("Loading configuration from: \(configPath)")
            // TODO: Phase 2 - Implement config file loading
        }

        // Phase 1: Model loading
        logger.info("📦 Loading model: \(model)")
        let loader = ModelLoader()

        do {
            let modelContainer = try await loader.load(modelPath: model)
            logger.info("✅ Model loaded successfully")

            // Phase 2: Initialize inference engine
            logger.info("🧠 Initializing inference engine...")
            let engine = InferenceEngine()
            await engine.initialize(modelContainer: modelContainer)
            logger.info("✅ Inference engine initialized")

            // Test mode: run single inference and exit
            if test {
                logger.info("🧪 Test mode: Running single inference...")
                try await runTestInference(engine: engine, logger: logger)
                logger.info("✅ Test completed successfully")
                return
            }

            // Phase 5: Start API server
            logger.info("🌐 Starting API server on port \(port)...")

            // Initialize scheduler and batcher
            logger.info("📋 Initializing request scheduler...")
            let scheduler = RequestScheduler()

            logger.info("🔄 Initializing continuous batcher...")
            let batcher = ContinuousBatcher(
                scheduler: scheduler,
                engine: engine,
                config: ContinuousBatcher.Config(maxBatchSize: 32, eosTokenId: 2)
            )

            // Start batching loop in background
            logger.info("▶️  Starting continuous batching loop...")
            Task {
                await batcher.start()
            }

            // Small delay to ensure batcher starts
            try await Task.sleep(for: .milliseconds(100))

            // Create and configure Vapor application
            let app = try await Application.make(.detect())

            do {
                // Configure server
                app.http.server.configuration.hostname = "0.0.0.0"
                app.http.server.configuration.port = port

                // Configure routes
                try routes(app, scheduler: scheduler, engine: engine, batcher: batcher)

                logger.info("✅ Server initialized successfully")
                logger.info("🚀 Server running on http://0.0.0.0:\(port)")
                logger.info("💡 Press Ctrl+C to stop the server")

                // Run server (blocks until shutdown)
                try await app.execute()
            } catch {
                try await app.asyncShutdown()
                throw error
            }

            try await app.asyncShutdown()

        } catch {
            logger.error("❌ Failed to start server: \(error.localizedDescription)")
            throw ExitCode.failure
        }
    }

    private func runTestInference(engine: InferenceEngine, logger: Logger) async throws {
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

    private func parseLogLevel(_ level: String) -> Logger.Level {
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
