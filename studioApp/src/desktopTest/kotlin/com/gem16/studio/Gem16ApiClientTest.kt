package com.gem16.studio

import com.gem16.studio.model.ChatMessage
import com.gem16.studio.model.GenerationConfig
import com.gem16.studio.model.MediaAttachment
import com.gem16.studio.model.MediaKind
import com.gem16.studio.model.ServerConfig
import com.gem16.studio.model.ThinkingEffort
import com.gem16.studio.service.ChatDelta
import com.gem16.studio.service.Gem16ApiClient
import com.sun.net.httpserver.HttpServer
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.net.InetSocketAddress
import java.util.concurrent.Executors
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class Gem16ApiClientTest {
    @Test
    fun streamsReasoningTextUsageSessionAndMultimodalContent() {
        runBlocking {
            val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
            server.executor = Executors.newSingleThreadExecutor()
            server.createContext("/v1/chat/completions") { exchange ->
                val request = Json.parseToJsonElement(
                    exchange.requestBody.bufferedReader().use { it.readText() },
                ).jsonObject
                assertEquals("gem16", request["model"]?.jsonPrimitive?.content)
                assertEquals("high", request["reasoning_effort"]?.jsonPrimitive?.content)
                assertTrue(request["stream"]?.jsonPrimitive?.content.toBoolean())
                val content = request["messages"]?.jsonArray?.first()?.jsonObject
                    ?.get("content")?.jsonArray
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
                ) { deltas += it }
                assertEquals("session_test", result.sessionId)
                assertEquals("stop", result.finishReason)
                assertEquals(12, result.usage?.promptTokens)
                assertEquals("think ", assertIs<ChatDelta.Reasoning>(deltas[0]).value)
                assertEquals("answer", assertIs<ChatDelta.Text>(deltas[1]).value)
                assertIs<ChatDelta.Finished>(deltas[2])
            } finally {
                server.stop(0)
                (server.executor as java.util.concurrent.ExecutorService).shutdownNow()
            }
        }
    }
}
