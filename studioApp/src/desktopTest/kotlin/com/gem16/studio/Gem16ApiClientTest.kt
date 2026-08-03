package com.gem16.studio

import com.gem16.studio.model.ChatMessage
import com.gem16.studio.model.GenerationConfig
import com.gem16.studio.model.MediaAttachment
import com.gem16.studio.model.MediaKind
import com.gem16.studio.model.ServerConfig
import com.gem16.studio.model.StreamPerformanceStats
import com.gem16.studio.model.ThinkingEffort
import com.gem16.studio.model.ToolCall
import com.gem16.studio.service.ChatDelta
import com.gem16.studio.service.ChatRequestPhase
import com.gem16.studio.service.Gem16ApiClient
import com.sun.net.httpserver.HttpServer
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.net.InetSocketAddress
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertTrue

class Gem16ApiClientTest {
    @Test
    fun streamsReasoningTextUsageSessionAndMultimodalContent() {
        runBlocking {
            val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
            server.executor = Executors.newSingleThreadExecutor()
            val metricsRequests = AtomicInteger()
            server.createContext("/metrics") { exchange ->
                val completed = metricsRequests.getAndIncrement() > 0
                val body = if (completed) {
                    """
                        gem16_input_tokens_total 112
                        gem16_cache_write_tokens_total 112
                        gem16_prompt_microseconds_total 6000
                        gem16_decode_microseconds_total 140000
                        gem16_decode_measured_tokens_total 7
                    """.trimIndent()
                } else {
                    """
                        gem16_input_tokens_total 100
                        gem16_cache_write_tokens_total 100
                        gem16_prompt_microseconds_total 0
                        gem16_decode_microseconds_total 0
                        gem16_decode_measured_tokens_total 0
                    """.trimIndent()
                }.toByteArray()
                exchange.sendResponseHeaders(200, body.size.toLong())
                exchange.responseBody.use { it.write(body) }
            }
            server.createContext("/v1/chat/completions") { exchange ->
                val request = Json.parseToJsonElement(
                    exchange.requestBody.bufferedReader().use { it.readText() },
                ).jsonObject
                assertEquals("gem16", request["model"]?.jsonPrimitive?.content)
                assertEquals("high", request["reasoning_effort"]?.jsonPrimitive?.content)
                assertTrue(request["stream"]?.jsonPrimitive?.content.toBoolean())
                val messages = request["messages"]?.jsonArray ?: error("messages are missing")
                assertEquals("system", messages.first().jsonObject["role"]?.jsonPrimitive?.content)
                assertEquals(
                    "You are a helpful assistant.",
                    messages.first().jsonObject["content"]?.jsonPrimitive?.content,
                )
                assertEquals(2, request["tools"]?.jsonArray?.size)
                val content = messages.first { it.jsonObject["role"]?.jsonPrimitive?.content == "user" }.jsonObject
                    .get("content")?.jsonArray
                    ?: error("multimodal content array is missing")
                assertEquals("text", content[0].jsonObject["type"]?.jsonPrimitive?.content)
                assertEquals("image_url", content[1].jsonObject["type"]?.jsonPrimitive?.content)
                assertEquals(
                    "data:image/png;base64,AQID",
                    content[1].jsonObject["image_url"]?.jsonObject?.get("url")?.jsonPrimitive?.content,
                )
                assertEquals("input_audio", content[2].jsonObject["type"]?.jsonPrimitive?.content)
                assertEquals(
                    "wav",
                    content[2].jsonObject["input_audio"]?.jsonObject?.get("format")?.jsonPrimitive?.content,
                )
                assertEquals(
                    "BAU=",
                    content[2].jsonObject["input_audio"]?.jsonObject?.get("data")?.jsonPrimitive?.content,
                )
                val body = listOf(
                    "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think \"},\"finish_reason\":null}]}",
                    "",
                    "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":null}]}",
                    "",
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_current_time\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}",
                    "",
                    "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{}\"}}]},\"finish_reason\":null}]}",
                    "",
                    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}",
                    "",
                    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7,\"total_tokens\":19}}",
                    "",
                    "data: [DONE]",
                    "",
                ).joinToString("\n").toByteArray()
                exchange.responseHeaders.add("Content-Type", "text/event-stream")
                exchange.responseHeaders.add("X-Gem16-Session-Id", "session_test")
                exchange.sendResponseHeaders(200, body.size.toLong())
                exchange.responseBody.use { it.write(body) }
            }
            server.start()
            try {
                val deltas = mutableListOf<ChatDelta>()
                val phases = mutableListOf<ChatRequestPhase>()
                val progress = mutableListOf<StreamPerformanceStats>()
                val result = Gem16ApiClient().streamChat(
                    server = ServerConfig(port = server.address.port),
                    generation = GenerationConfig(
                        thinking = ThinkingEffort.High,
                        maxOutputTokens = 128,
                    ),
                    messages = listOf(
                        ChatMessage(
                            role = "user",
                            content = "describe and transcribe",
                            attachments = listOf(
                                MediaAttachment(
                                    fileName = "pixel.png",
                                    kind = MediaKind.Image,
                                    mimeType = "image/png",
                                    format = "png",
                                    bytes = byteArrayOf(1, 2, 3),
                                ),
                                MediaAttachment(
                                    fileName = "clip.wav",
                                    kind = MediaKind.Audio,
                                    mimeType = "audio/wav",
                                    format = "wav",
                                    bytes = byteArrayOf(4, 5),
                                ),
                            ),
                        ),
                    ),
                    sessionId = null,
                    onPhase = { phases += it },
                    onProgress = { progress += it },
                    onDelta = { deltas += it },
                )
                assertEquals("session_test", result.sessionId)
                assertEquals("stop", result.finishReason)
                assertEquals(12, result.usage?.promptTokens)
                assertEquals("get_current_time", result.toolCalls.single().name)
                assertEquals("{}", result.toolCalls.single().argumentsJson)
                assertEquals(50.0, result.performance?.decodeTokensPerSecond)
                assertEquals(2000.0, result.performance?.prefillTokensPerSecond)
                assertEquals(6.0, result.performance?.prefillMilliseconds)
                assertEquals(listOf(1L, 2L), progress.map(StreamPerformanceStats::emittedTokens))
                assertEquals(
                    listOf(
                        ChatRequestPhase.PreparingRequest,
                        ChatRequestPhase.WaitingForFirstToken,
                        ChatRequestPhase.Decoding,
                    ),
                    phases,
                )
                assertTrue(progress.all { it.firstTokenMilliseconds >= 0.0 })
                assertTrue(progress.last().tokensPerSecond != null)
                assertEquals("think ", assertIs<ChatDelta.Reasoning>(deltas[0]).value)
                assertEquals("answer", assertIs<ChatDelta.Text>(deltas[1]).value)
                assertIs<ChatDelta.Finished>(deltas[2])
            } finally {
                server.stop(0)
                (server.executor as java.util.concurrent.ExecutorService).shutdownNow()
            }
        }
    }

    @Test
    fun serializesDocumentsAndToolRoundHistoryAsOpenAiMessages() {
        runBlocking {
            val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
            server.executor = Executors.newSingleThreadExecutor()
            server.createContext("/v1/chat/completions") { exchange ->
                val request = Json.parseToJsonElement(
                    exchange.requestBody.bufferedReader().use { it.readText() },
                ).jsonObject
                val messages = request["messages"]?.jsonArray ?: error("messages are missing")
                assertEquals(listOf("system", "user", "assistant", "tool", "assistant", "user"), messages.map {
                    it.jsonObject["role"]?.jsonPrimitive?.content
                })
                val userText = messages[1].jsonObject["content"]?.jsonPrimitive?.content.orEmpty()
                assertTrue(userText.contains("Begin attached document: notes.txt"))
                assertTrue(userText.contains("Document body"))
                assertEquals(
                    "get_current_date",
                    messages[2].jsonObject["tool_calls"]?.jsonArray?.single()?.jsonObject
                        ?.get("function")?.jsonObject?.get("name")?.jsonPrimitive?.content,
                )
                assertEquals("call_date", messages[3].jsonObject["tool_call_id"]?.jsonPrimitive?.content)
                assertEquals(
                    "It is Monday.",
                    messages[4].jsonObject["content"]?.jsonPrimitive?.content,
                )
                assertTrue(
                    messages[5].jsonObject["content"]?.jsonPrimitive?.content.orEmpty()
                        .contains("Begin attached document: thesis.pdf"),
                )
                val body = listOf(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"It is Monday.\"},\"finish_reason\":null}]}",
                    "",
                    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}",
                    "",
                    "data: [DONE]",
                    "",
                ).joinToString("\n").toByteArray()
                exchange.responseHeaders.add("Content-Type", "text/event-stream")
                exchange.sendResponseHeaders(200, body.size.toLong())
                exchange.responseBody.use { it.write(body) }
            }
            server.start()
            try {
                val result = Gem16ApiClient().streamChat(
                    server = ServerConfig(port = server.address.port),
                    generation = GenerationConfig(),
                    messages = listOf(
                        ChatMessage(
                            role = "user",
                            content = "Summarize this.",
                            attachments = listOf(
                                MediaAttachment(
                                    fileName = "notes.txt",
                                    kind = MediaKind.Document,
                                    mimeType = "text/plain",
                                    format = "txt",
                                    bytes = ByteArray(0),
                                    documentText = "Document body",
                                    sourceByteSize = 13,
                                ),
                            ),
                        ),
                        ChatMessage(
                            role = "assistant",
                            content = "",
                            toolCalls = listOf(ToolCall("call_date", "get_current_date", "{}")),
                        ),
                        ChatMessage(role = "tool", content = "{\"date\":\"2026-08-03\"}", toolCallId = "call_date"),
                        ChatMessage(role = "assistant", content = " \nIt is Monday.\r\n"),
                        ChatMessage(
                            role = "user",
                            content = "Analyze this.",
                            attachments = listOf(
                                MediaAttachment(
                                    fileName = "thesis.pdf",
                                    kind = MediaKind.Document,
                                    mimeType = "application/pdf",
                                    format = "pdf",
                                    bytes = ByteArray(0),
                                    documentText = "Thesis body",
                                    sourceByteSize = 1024,
                                ),
                            ),
                        ),
                    ),
                    sessionId = "session_tools",
                    onDelta = {},
                )
                assertEquals("stop", result.finishReason)
                assertTrue(result.toolCalls.isEmpty())
            } finally {
                server.stop(0)
                (server.executor as java.util.concurrent.ExecutorService).shutdownNow()
            }
        }
    }

    @Test
    fun surfacesStreamingErrorsAndAbruptlyEndedStreams() {
        runBlocking {
            val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
            server.executor = Executors.newSingleThreadExecutor()
            val requestCount = AtomicInteger()
            server.createContext("/v1/chat/completions") { exchange ->
                exchange.requestBody.close()
                val body = if (requestCount.getAndIncrement() == 0) {
                    "data: {\"error\":{\"message\":\"prompt exceeds the context capacity\",\"type\":\"server_error\"}}\n\n"
                } else {
                    "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"stop\"}]}\n\n"
                }.toByteArray()
                exchange.responseHeaders.add("Content-Type", "text/event-stream")
                exchange.sendResponseHeaders(200, body.size.toLong())
                exchange.responseBody.use { it.write(body) }
            }
            server.start()
            try {
                val client = Gem16ApiClient()
                val config = ServerConfig(port = server.address.port)
                val messages = listOf(ChatMessage(role = "user", content = "hello"))
                val streamError = assertFailsWith<IllegalStateException> {
                    client.streamChat(config, GenerationConfig(), messages, null, onDelta = {})
                }
                assertTrue(streamError.message?.contains("prompt exceeds the context capacity") == true)
                val abruptEnd = assertFailsWith<IllegalStateException> {
                    client.streamChat(config, GenerationConfig(), messages, null, onDelta = {})
                }
                assertTrue(abruptEnd.message?.contains("without a [DONE] marker") == true)
            } finally {
                server.stop(0)
                (server.executor as java.util.concurrent.ExecutorService).shutdownNow()
            }
        }
    }
}
