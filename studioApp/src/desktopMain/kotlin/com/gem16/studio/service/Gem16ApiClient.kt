package com.gem16.studio.service

import com.gem16.studio.model.ChatMessage
import com.gem16.studio.model.GenerationConfig
import com.gem16.studio.model.MediaKind
import com.gem16.studio.model.PerformanceStats
import com.gem16.studio.model.ServerConfig
import com.gem16.studio.model.StreamPerformanceStats
import com.gem16.studio.model.ToolCall
import com.gem16.studio.model.Usage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put
import java.io.InputStream
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.time.Duration
import java.util.Base64
import java.util.concurrent.atomic.AtomicReference

sealed interface ChatDelta {
    data class Text(val value: String) : ChatDelta
    data class Reasoning(val value: String) : ChatDelta
    data class Finished(val reason: String?, val usage: Usage?) : ChatDelta
}

enum class ChatRequestPhase {
    PreparingRequest,
    WaitingForFirstToken,
    Decoding,
}

data class ChatStreamResult(
    val sessionId: String?,
    val usage: Usage?,
    val finishReason: String?,
    val performance: PerformanceStats?,
    val toolCalls: List<ToolCall>,
)

private data class ToolCallAccumulator(
    var id: String = "",
    var name: String = "",
    val arguments: StringBuilder = StringBuilder(),
)

private data class ServerMetrics(
    val inputTokens: Double,
    val cacheWriteTokens: Double,
    val promptMicroseconds: Double,
    val decodeMicroseconds: Double,
    val decodeMeasuredTokens: Double,
)

class Gem16ApiClient {
    private val json = Json { ignoreUnknownKeys = true }
    private val http = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(10))
        .version(HttpClient.Version.HTTP_1_1)
        .build()
    private val activeStream = AtomicReference<InputStream?>(null)

    suspend fun streamChat(
        server: ServerConfig,
        generation: GenerationConfig,
        messages: List<ChatMessage>,
        sessionId: String?,
        onPhase: suspend (ChatRequestPhase) -> Unit = {},
        onProgress: suspend (StreamPerformanceStats) -> Unit = {},
        onDelta: suspend (ChatDelta) -> Unit,
    ): ChatStreamResult = withContext(Dispatchers.IO) {
        onPhase(ChatRequestPhase.PreparingRequest)
        val metricsBefore = fetchMetrics(server)
        val requestStartedNanos = System.nanoTime()
        var firstTokenNanos: Long? = null
        var emittedTokens = 0L
        suspend fun reportProgress() {
            val now = System.nanoTime()
            val first = firstTokenNanos ?: now.also { firstTokenNanos = it }
            emittedTokens += 1L
            val decodeNanos = now - first
            onProgress(
                StreamPerformanceStats(
                    emittedTokens = emittedTokens,
                    tokensPerSecond = if (emittedTokens > 1L && decodeNanos > 0L) {
                        (emittedTokens - 1L) * 1_000_000_000.0 / decodeNanos
                    } else {
                        null
                    },
                    firstTokenMilliseconds = (first - requestStartedNanos) / 1_000_000.0,
                    elapsedMilliseconds = (now - requestStartedNanos) / 1_000_000.0,
                ),
            )
        }
        val payload = requestPayload(server, generation, messages)
        val builder = HttpRequest.newBuilder(URI.create("${server.baseUrl}/chat/completions"))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("Authorization", "Bearer gem16")
            .POST(HttpRequest.BodyPublishers.ofString(json.encodeToString(JsonObject.serializer(), payload)))
        if (!sessionId.isNullOrBlank()) builder.header("X-Gem16-Session-Id", sessionId)
        val response = http.send(builder.build(), HttpResponse.BodyHandlers.ofInputStream())
        if (response.statusCode() !in 200..299) {
            val detail = response.body().bufferedReader().use { it.readText() }
            throw IllegalStateException("Server returned HTTP ${response.statusCode()}: ${errorMessage(detail)}")
        }
        onPhase(ChatRequestPhase.WaitingForFirstToken)
        val stream = response.body()
        activeStream.set(stream)
        var usage: Usage? = null
        var finishReason: String? = null
        val toolCalls = sortedMapOf<Int, ToolCallAccumulator>()
        var sawDone = false
        var decodingReported = false
        var receivedModelOutput = false
        suspend fun reportDecoding() {
            if (!decodingReported) {
                decodingReported = true
                onPhase(ChatRequestPhase.Decoding)
            }
        }
        try {
            stream.bufferedReader(Charsets.UTF_8).useLines { lines ->
                for (line in lines) {
                    if (!line.startsWith("data:")) continue
                    val data = line.removePrefix("data:").trim()
                    if (data.isEmpty()) continue
                    if (data == "[DONE]") {
                        sawDone = true
                        break
                    }
                    val root = json.parseToJsonElement(data).jsonObject
                    root["error"]?.takeUnless { it is JsonNull }?.jsonObject?.let { error ->
                        val message = error["message"]?.jsonPrimitive?.contentOrNull
                            ?: "The server reported a streaming error"
                        throw IllegalStateException("Server stream error: $message")
                    }
                    parseUsage(root)?.let { usage = it }
                    val choice = root["choices"]?.jsonArray?.firstOrNull()?.jsonObject ?: continue
                    choice["finish_reason"]?.takeUnless { it is JsonNull }
                        ?.jsonPrimitive?.contentOrNull?.let { finishReason = it }
                    val delta = choice["delta"]?.jsonObject ?: continue
                    delta["tool_calls"]?.jsonArray?.forEach { value ->
                        receivedModelOutput = true
                        reportDecoding()
                        val chunk = value.jsonObject
                        val index = chunk["index"]?.jsonPrimitive?.intOrNull
                            ?: throw IllegalStateException("A streamed tool call is missing its index")
                        val accumulator = toolCalls.getOrPut(index) { ToolCallAccumulator() }
                        chunk["id"]?.jsonPrimitive?.contentOrNull?.let { accumulator.id = it }
                        chunk["function"]?.jsonObject?.let { function ->
                            function["name"]?.jsonPrimitive?.contentOrNull?.let { accumulator.name = it }
                            function["arguments"]?.jsonPrimitive?.contentOrNull
                                ?.let { accumulator.arguments.append(it) }
                        }
                    }
                    delta["reasoning_content"]?.jsonPrimitive?.contentOrNull
                        ?.takeIf(String::isNotEmpty)
                        ?.let {
                            receivedModelOutput = true
                            reportDecoding()
                            reportProgress()
                            onDelta(ChatDelta.Reasoning(it))
                        }
                    delta["content"]?.jsonPrimitive?.contentOrNull
                        ?.takeIf(String::isNotEmpty)
                        ?.let {
                            receivedModelOutput = true
                            reportDecoding()
                            reportProgress()
                            onDelta(ChatDelta.Text(it))
                        }
                }
            }
        } finally {
            activeStream.compareAndSet(stream, null)
            runCatching { stream.close() }
        }
        check(sawDone) { "The server stream ended without a [DONE] marker" }
        check(finishReason != null) { "The server stream ended without a finish reason" }
        check(receivedModelOutput) { "The server completed without returning any model output" }
        onDelta(ChatDelta.Finished(finishReason, usage))
        val performance = metricsBefore?.let { before ->
            fetchMetrics(server)?.let { after -> performanceDifference(before, after) }
        }
        ChatStreamResult(
            sessionId = response.headers().firstValue("X-Gem16-Session-Id").orElse(sessionId),
            usage = usage,
            finishReason = finishReason,
            performance = performance,
            toolCalls = toolCalls.map { (_, call) ->
                check(call.id.isNotBlank() && call.name.isNotBlank()) { "The server returned an incomplete tool call" }
                ToolCall(call.id, call.name, call.arguments.toString())
            },
        )
    }

    fun cancelActive() {
        activeStream.getAndSet(null)?.let { runCatching { it.close() } }
    }

    private fun requestPayload(
        server: ServerConfig,
        generation: GenerationConfig,
        messages: List<ChatMessage>,
    ): JsonObject = buildJsonObject {
        put("model", server.modelName)
        put("stream", true)
        put("max_completion_tokens", generation.maxOutputTokens)
        put("reasoning_effort", generation.thinking.wireValue)
        put("stream_options", buildJsonObject { put("include_usage", true) })
        if (generation.localDateTimeTools) {
            put("tools", localDateTimeTools())
            put("parallel_tool_calls", true)
        }
        put("messages", buildJsonArray {
            generation.systemPrompt.takeIf(String::isNotBlank)?.let { prompt ->
                add(buildJsonObject {
                    put("role", "system")
                    put("content", prompt)
                })
            }
            messages.filter { it.role == "user" || it.role == "assistant" || it.role == "tool" }
                .forEach { message ->
                add(buildJsonObject {
                    put("role", message.role)
                    if (message.role == "tool") {
                        put("content", message.content)
                        put("tool_call_id", requireNotNull(message.toolCallId))
                    } else if (message.role == "assistant") {
                        if (message.content.isBlank() && message.toolCalls.isNotEmpty()) {
                            put("content", JsonNull)
                        } else {
                            // The server stores generated assistant text using
                            // its response-level trim, whereas streaming exposes
                            // token bytes before trailing whitespace is known.
                            // Normalize the history sent back to a resident session.
                            put("content", message.content.trim())
                        }
                        if (message.toolCalls.isNotEmpty()) {
                            put("tool_calls", buildJsonArray {
                                message.toolCalls.forEach { call ->
                                    add(buildJsonObject {
                                        put("id", call.id)
                                        put("type", "function")
                                        put("function", buildJsonObject {
                                            put("name", call.name)
                                            put("arguments", call.argumentsJson)
                                        })
                                    })
                                }
                            })
                        }
                    } else {
                        val text = userTextWithDocuments(message)
                        val media = message.attachments.filter { it.kind != MediaKind.Document }
                        if (media.isEmpty()) {
                            put("content", text)
                        } else {
                            put("content", buildJsonArray {
                                if (text.isNotBlank()) {
                                    add(buildJsonObject {
                                        put("type", "text")
                                        put("text", text)
                                    })
                                }
                                media.forEach { attachment ->
                                    val encoded = Base64.getEncoder().encodeToString(attachment.bytes)
                                    add(
                                        when (attachment.kind) {
                                            MediaKind.Image -> buildJsonObject {
                                                put("type", "image_url")
                                                put("image_url", buildJsonObject {
                                                    put(
                                                        "url",
                                                        "data:${attachment.mimeType};base64,$encoded",
                                                    )
                                                })
                                            }
                                            MediaKind.Audio -> buildJsonObject {
                                                put("type", "input_audio")
                                                put("input_audio", buildJsonObject {
                                                    put("format", attachment.format)
                                                    put("data", encoded)
                                                })
                                            }
                                            MediaKind.Document -> error(
                                                "Document attachments must be rendered as text",
                                            )
                                        },
                                    )
                                }
                            })
                        }
                    }
                })
            }
        })
    }

    private fun userTextWithDocuments(message: ChatMessage): String = buildString {
        if (message.content.isNotBlank()) append(message.content.trim())
        message.attachments.filter { it.kind == MediaKind.Document }.forEach { attachment ->
            if (isNotEmpty()) append("\n\n")
            val safeName = attachment.fileName.replace('\n', ' ').replace('\r', ' ')
            append("--- Begin attached document: ").append(safeName).append(" ---\n")
            append(requireNotNull(attachment.documentText))
            append("\n--- End attached document: ").append(safeName).append(" ---")
        }
    }

    private fun localDateTimeTools(): JsonArray = buildJsonArray {
        add(localTool(
            name = "get_current_date",
            description = "Get the current local calendar date and timezone from the user's computer.",
        ))
        add(localTool(
            name = "get_current_time",
            description = "Get the current local system time, UTC offset, timezone, and ISO date-time from the user's computer.",
        ))
    }

    private fun localTool(name: String, description: String): JsonObject = buildJsonObject {
        put("type", "function")
        put("function", buildJsonObject {
            put("name", name)
            put("description", description)
            put("strict", true)
            put("parameters", buildJsonObject {
                put("type", "object")
                put("properties", buildJsonObject {})
                put("additionalProperties", false)
            })
        })
    }

    private fun parseUsage(root: JsonObject): Usage? {
        val value = root["usage"]?.takeUnless { it is JsonNull }?.jsonObject ?: return null
        val prompt = value["prompt_tokens"]?.jsonPrimitive?.longOrNull ?: 0
        val completion = value["completion_tokens"]?.jsonPrimitive?.longOrNull ?: 0
        val total = value["total_tokens"]?.jsonPrimitive?.longOrNull ?: prompt + completion
        return Usage(prompt, completion, total)
    }

    private fun fetchMetrics(server: ServerConfig): ServerMetrics? = runCatching {
        val request = HttpRequest.newBuilder(URI.create("http://${server.clientHost}:${server.port}/metrics"))
            .timeout(Duration.ofSeconds(3))
            .GET()
            .build()
        val response = http.send(request, HttpResponse.BodyHandlers.ofString())
        if (response.statusCode() !in 200..299) return@runCatching null
        parseServerMetrics(response.body())
    }.getOrNull()

    private fun errorMessage(body: String): String = runCatching {
        json.parseToJsonElement(body).jsonObject["error"]?.jsonObject
            ?.get("message")?.jsonPrimitive?.contentOrNull
    }.getOrNull() ?: body.take(500)
}

private fun parseServerMetrics(body: String): ServerMetrics? {
    val values = body.lineSequence().mapNotNull { line ->
        val clean = line.trim()
        if (clean.isEmpty() || clean.startsWith('#')) return@mapNotNull null
        val separator = clean.indexOfAny(charArrayOf(' ', '\t'))
        if (separator <= 0) return@mapNotNull null
        val name = clean.substring(0, separator)
        val value = clean.substring(separator).trim().toDoubleOrNull() ?: return@mapNotNull null
        name to value
    }.toMap()
    fun metric(name: String): Double? = values[name]
    return ServerMetrics(
        inputTokens = metric("gem16_input_tokens_total") ?: return null,
        cacheWriteTokens = metric("gem16_cache_write_tokens_total") ?: return null,
        promptMicroseconds = metric("gem16_prompt_microseconds_total") ?: return null,
        decodeMicroseconds = metric("gem16_decode_microseconds_total") ?: return null,
        decodeMeasuredTokens = metric("gem16_decode_measured_tokens_total") ?: return null,
    )
}

private fun performanceDifference(before: ServerMetrics, after: ServerMetrics): PerformanceStats? {
    val promptMicros = after.promptMicroseconds - before.promptMicroseconds
    val decodeMicros = after.decodeMicroseconds - before.decodeMicroseconds
    val inputTokens = after.inputTokens - before.inputTokens
    val cacheWriteTokens = after.cacheWriteTokens - before.cacheWriteTokens
    val decodeTokens = after.decodeMeasuredTokens - before.decodeMeasuredTokens
    if (promptMicros < 0.0 || decodeMicros <= 0.0 || inputTokens < 0.0 || decodeTokens < 0.0) return null
    return PerformanceStats(
        decodeTokensPerSecond = decodeTokens * 1_000_000.0 / decodeMicros,
        prefillTokensPerSecond = if (promptMicros > 0.0) {
            cacheWriteTokens * 1_000_000.0 / promptMicros
        } else {
            0.0
        },
        prefillMilliseconds = promptMicros / 1_000.0,
        decodeMilliseconds = decodeMicros / 1_000.0,
    )
}
