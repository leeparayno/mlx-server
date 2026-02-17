import XCTest
@testable import Scheduler

final class TokenStreamTests: XCTestCase {
    func testTokenChunkCreation() {
        let requestId = UUID()
        let chunk = TokenChunk(
            requestId: requestId,
            token: "hello",
            index: 0
        )

        XCTAssertEqual(chunk.requestId, requestId)
        XCTAssertEqual(chunk.token, "hello")
        XCTAssertEqual(chunk.index, 0)
        XCTAssertNotNil(chunk.timestamp)
    }

    func testGenerationInfo() {
        let requestId = UUID()
        let info = GenerationInfo(
            requestId: requestId,
            totalTokens: 100,
            duration: 10.0,
            finishReason: .stop
        )

        XCTAssertEqual(info.requestId, requestId)
        XCTAssertEqual(info.totalTokens, 100)
        XCTAssertEqual(info.duration, 10.0)
        XCTAssertEqual(info.tokensPerSecond, 10.0)
        XCTAssertEqual(info.finishReason, .stop)
    }

    func testGenerationInfoZeroDuration() {
        let info = GenerationInfo(
            requestId: UUID(),
            totalTokens: 100,
            duration: 0.0,
            finishReason: .stop
        )

        XCTAssertEqual(info.tokensPerSecond, 0)
    }

    func testTokenStreamRegistryRegister() async {
        let registry = TokenStreamRegistry()
        let requestId = UUID()

        let stream = await registry.register(requestId: requestId)
        XCTAssertNotNil(stream)

        let isRegistered = await registry.isRegistered(requestId: requestId)
        XCTAssertTrue(isRegistered)

        let count = await registry.activeStreamCount
        XCTAssertEqual(count, 1)
    }

    func testTokenStreamRegistryYield() async {
        let registry = TokenStreamRegistry()
        let requestId = UUID()

        let stream = await registry.register(requestId: requestId)

        // Start consuming stream in background
        let expectation = expectation(description: "Stream receives token")

        // Use actor to safely store received token
        actor TokenHolder {
            var token: String?

            func setToken(_ token: String) {
                self.token = token
            }

            func getToken() -> String? {
                token
            }
        }

        let tokenHolder = TokenHolder()

        Task {
            do {
                for try await chunk in stream {
                    await tokenHolder.setToken(chunk.token)
                    expectation.fulfill()
                    break
                }
            } catch {
                XCTFail("Stream threw error: \(error)")
            }
        }

        // Yield a token
        let chunk = TokenChunk(requestId: requestId, token: "test", index: 0)
        await registry.yield(requestId: requestId, chunk: chunk)

        await fulfillment(of: [expectation], timeout: 1.0)

        let receivedToken = await tokenHolder.getToken()
        XCTAssertEqual(receivedToken, "test")
    }

    func testTokenStreamRegistryFinish() async {
        let registry = TokenStreamRegistry()
        let requestId = UUID()

        let stream = await registry.register(requestId: requestId)

        let expectation = expectation(description: "Stream finishes")

        // Use actor to safely store stream ended flag
        actor FlagHolder {
            var streamEnded = false

            func setEnded() {
                streamEnded = true
            }

            func getEnded() -> Bool {
                streamEnded
            }
        }

        let flagHolder = FlagHolder()

        Task {
            do {
                for try await _ in stream {
                    // No tokens expected
                }
                await flagHolder.setEnded()
                expectation.fulfill()
            } catch {
                XCTFail("Stream threw error: \(error)")
            }
        }

        // Finish the stream
        await registry.finish(requestId: requestId)

        await fulfillment(of: [expectation], timeout: 1.0)

        let streamEnded = await flagHolder.getEnded()
        XCTAssertTrue(streamEnded)

        // Should be unregistered
        let isRegistered = await registry.isRegistered(requestId: requestId)
        XCTAssertFalse(isRegistered)
    }

    func testTokenStreamRegistryFinishWithError() async {
        let registry = TokenStreamRegistry()
        let requestId = UUID()

        let stream = await registry.register(requestId: requestId)

        let expectation = expectation(description: "Stream receives error")

        // Use actor to safely store received error
        actor ErrorHolder {
            var error: StreamError?

            func setError(_ error: StreamError) {
                self.error = error
            }

            func getError() -> StreamError? {
                error
            }
        }

        let errorHolder = ErrorHolder()

        Task {
            do {
                for try await _ in stream {
                    // No tokens expected
                }
                XCTFail("Should have thrown error")
            } catch let error as StreamError {
                await errorHolder.setError(error)
                expectation.fulfill()
            } catch {
                XCTFail("Wrong error type: \(error)")
            }
        }

        // Finish with error
        await registry.finish(requestId: requestId, error: StreamError.timeout)

        await fulfillment(of: [expectation], timeout: 1.0)

        let receivedError = await errorHolder.getError()
        XCTAssertEqual(receivedError, StreamError.timeout)

        // Should be unregistered
        let isRegistered = await registry.isRegistered(requestId: requestId)
        XCTAssertFalse(isRegistered)
    }

    func testTokenStreamMultipleTokens() async {
        let registry = TokenStreamRegistry()
        let requestId = UUID()

        let stream = await registry.register(requestId: requestId)

        let expectation = expectation(description: "Stream receives all tokens")

        // Use actor to safely store received tokens
        actor TokenHolder {
            var tokens: [String] = []

            func addToken(_ token: String) {
                tokens.append(token)
            }

            func getTokens() -> [String] {
                tokens
            }
        }

        let tokenHolder = TokenHolder()

        Task {
            do {
                for try await chunk in stream {
                    await tokenHolder.addToken(chunk.token)
                    let count = await tokenHolder.getTokens().count
                    if count == 3 {
                        expectation.fulfill()
                    }
                }
            } catch {
                XCTFail("Stream threw error: \(error)")
            }
        }

        // Yield multiple tokens
        for i in 0..<3 {
            let chunk = TokenChunk(requestId: requestId, token: "token\(i)", index: i)
            await registry.yield(requestId: requestId, chunk: chunk)
        }

        await fulfillment(of: [expectation], timeout: 1.0)

        let receivedTokens = await tokenHolder.getTokens()
        XCTAssertEqual(receivedTokens, ["token0", "token1", "token2"])
    }

    func testTokenStreamYieldToUnregistered() async {
        let registry = TokenStreamRegistry()
        let requestId = UUID()

        // Yield to non-existent stream (should not crash)
        let chunk = TokenChunk(requestId: requestId, token: "test", index: 0)
        await registry.yield(requestId: requestId, chunk: chunk)

        // Should still have zero streams
        let count = await registry.activeStreamCount
        XCTAssertEqual(count, 0)
    }

    func testTokenStreamCancelAll() async {
        let registry = TokenStreamRegistry()

        // Register multiple streams
        let id1 = UUID()
        let id2 = UUID()

        let stream1 = await registry.register(requestId: id1)
        let stream2 = await registry.register(requestId: id2)

        var count = await registry.activeStreamCount
        XCTAssertEqual(count, 2)

        let expectation1 = expectation(description: "Stream 1 cancelled")
        let expectation2 = expectation(description: "Stream 2 cancelled")

        Task {
            do {
                for try await _ in stream1 {
                    XCTFail("Should not receive tokens")
                }
            } catch {
                expectation1.fulfill()
            }
        }

        Task {
            do {
                for try await _ in stream2 {
                    XCTFail("Should not receive tokens")
                }
            } catch {
                expectation2.fulfill()
            }
        }

        // Cancel all
        await registry.cancelAll()

        await fulfillment(of: [expectation1, expectation2], timeout: 1.0)

        count = await registry.activeStreamCount
        XCTAssertEqual(count, 0)
    }
}

// MARK: - StreamError Equatable

extension StreamError: Equatable {
    public static func == (lhs: StreamError, rhs: StreamError) -> Bool {
        switch (lhs, rhs) {
        case (.streamClosed, .streamClosed),
             (.timeout, .timeout),
             (.cancelled, .cancelled):
            return true
        case let (.engineError(lhsMsg), .engineError(rhsMsg)):
            return lhsMsg == rhsMsg
        default:
            return false
        }
    }
}
