package com.gem16.studio.service

import com.gem16.studio.model.ChatMessage
import com.gem16.studio.model.GenerationConfig
import com.gem16.studio.model.MediaKind
import com.gem16.studio.model.ServerConfig
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

data class ChatStreamResult(
    val sessionId: String?,
    val usage: Usage?,
    val finishReason: String?,
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
        onDelta: suspend (ChatDelta) -> Unit,
    ): ChatStreamResult = withContext(Dispatchers.IO) {
        val payload = requestPayload(server, generation, messages)
        val builder = HttpRequest.newBuilder(URI.create("${server.baseUrl}/chat/completions"))
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream")
            .header("Authorization", "Bearer gem16-studio")
            .POST(HttpRequest.BodyPublishers.ofString(json.encodeToString(JsonObject.serializer(), payload)))
        if (!sessionId.isNullOrBlank()) builder.header("X-Gem16-Session-Id", sessionId)
        val response = http.send(builder.build(), HttpResponse.BodyHandlers.ofInputStream())
        if (response.statusCode() !in 200..299) {
            val detail = response.body().bufferedReader().use { it.readText() }
            throw IllegalStateException("Server returned HTTP ${response.statusCode()}: ${errorMessage(detail)}")
        }
        val stream = response.body()
        activeStream.set(stream)
        var usage: Usage? = null
        var finishReason: String? = null
        try {
            stream.bufferedReader(Charsets.UTF_8).useLines { lines ->
                for (line in lines) {
                    if (!line.startsWith("data:")) continue
                    val data = line.removePrefix("data:").trim()
                    if (data.isEmpty() || data == "[DONE]") continue
                    val root = json.parseToJsonElement(data).jsonObject
                    parseUsage(root)?.let { usage = it }
                    val choice = root["choices"]?.jsonArray?.firstOrNull()?.jsonObject ?: continue
                    choice["finish_reason"]?.takeUnless { it is JsonNull }
                        ?.jsonPrimitive?.contentOrNull?.let { finishReason = it }
                    val delta = choice["delta"]?.jsonObject ?: continue
                    delta["reasoning_content"]?.jsonPrimitive?.contentOrNull
                        ?.takeIf(String::isNotEmpty)
                        ?.let { onDelta(ChatDelta.Reasoning(it)) }
                    delta["content"]?.jsonPrimitive?.contentOrNull
                        ?.takeIf(String::isNotEmpty)
                        ?.let { onDelta(ChatDelta.Text(it)) }
                }
            }
        } finally {
            activeStream.compareAndSet(stream, null)
            runCatching { stream.close() }
        }
        onDelta(ChatDelta.Finished(finishReason, usage))
        ChatStreamResult(
            sessionId = response.headers().firstValue("X-Gem16-Session-Id").orElse(sessionId),
            usage = usage,
            finishReason = finishReason,
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
        put("messages", buildJsonArray {
            messages.filter { it.role == "user" || it.role == "assistant" }.forEach { message ->
                add(buildJsonObject {
                    put("role", message.role)
                    if (message.attachments.isEmpty()) {
                        put("content", message.content)
                    } else {
                        put("content", buildJsonArray {
                            if (message.content.isNotBlank()) {
                                add(buildJsonObject {
                                    put("type", "text")
                                    put("text", message.content)
                                })
                            }
                            message.attachments.forEach { attachment ->
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
                                    },
                                )
                            }
                        })
                    }
                })
            }
        })
    }

    private fun parseUsage(root: JsonObject): Usage? {
        val value = root["usage"]?.takeUnless { it is JsonNull }?.jsonObject ?: return null
        val prompt = value["prompt_tokens"]?.jsonPrimitive?.longOrNull ?: 0
        val completion = value["completion_tokens"]?.jsonPrimitive?.longOrNull ?: 0
        val total = value["total_tokens"]?.jsonPrimitive?.longOrNull ?: prompt + completion
        return Usage(prompt, completion, total)
    }

    private fun errorMessage(body: String): String = runCatching {
        json.parseToJsonElement(body).jsonObject["error"]?.jsonObject
            ?.get("message")?.jsonPrimitive?.contentOrNull
    }.getOrNull() ?: body.take(500)
}
