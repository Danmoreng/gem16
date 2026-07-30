package com.gem16.studio.model

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.util.UUID

enum class ServerPhase {
    Stopped,
    Starting,
    Running,
    Stopping,
    External,
    Error,
}

enum class MediaKind {
    Image,
    Audio,
}

data class MediaAttachment(
    val id: String = UUID.randomUUID().toString(),
    val fileName: String,
    val kind: MediaKind,
    val mimeType: String,
    val format: String,
    val bytes: ByteArray,
    val durationMillis: Long? = null,
) {
    val byteSize: Int get() = bytes.size
    val encodedSize: Long get() = 4L * ((bytes.size.toLong() + 2L) / 3L)
}

enum class ThinkingEffort(val wireValue: String, val label: String) {
    Off("none", "Off"),
    Low("low", "Low · 1K"),
    Medium("medium", "Medium · 4K"),
    High("high", "High · 8K"),
}

data class ServerConfig(
    val executable: String = defaultServerExecutable(),
    val modelDirectory: String = defaultModelDirectory(),
    val assistantModelDirectory: String = defaultAssistantDirectory(),
    val modelName: String = "gem16",
    val host: String = "127.0.0.1",
    val port: Int = 8080,
    val maxContextTokens: Long = 32768,
    val maxSessions: Int = 1,
    val mtpDraftTokens: Int = 2,
    val mtpAdaptive: Boolean = false,
    val greedy: Boolean = false,
) {
    val clientHost: String
        get() = when (host) {
            "0.0.0.0" -> "127.0.0.1"
            "::", "[::]" -> "[::1]"
            else -> host
        }
    val baseUrl: String get() = "http://$clientHost:$port/v1"
}

data class GenerationConfig(
    val thinking: ThinkingEffort = ThinkingEffort.Medium,
    val maxOutputTokens: Long = 16384,
    val showReasoning: Boolean = true,
)

data class StudioSettings(
    val server: ServerConfig = ServerConfig(),
    val generation: GenerationConfig = GenerationConfig(),
    val darkTheme: Boolean = true,
)

data class HealthSnapshot(
    val status: String,
    val residentSessions: Int,
    val sessionLimit: Int,
    val maxContextTokens: Long,
    val mtpDraftTokens: Int,
    val samplingEnabled: Boolean,
    val temperature: Double?,
    val topK: Int?,
    val topP: Double?,
)

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val role: String,
    val content: String,
    val attachments: List<MediaAttachment> = emptyList(),
    val reasoning: String = "",
    val streaming: Boolean = false,
    val error: String? = null,
)

data class Usage(
    val promptTokens: Long = 0,
    val completionTokens: Long = 0,
    val totalTokens: Long = 0,
)

data class PerformanceStats(
    val decodeTokensPerSecond: Double,
    val prefillTokensPerSecond: Double,
    val prefillMilliseconds: Double,
    val decodeMilliseconds: Double,
)

data class StreamPerformanceStats(
    val emittedTokens: Long,
    val tokensPerSecond: Double?,
    val firstTokenMilliseconds: Double,
    val elapsedMilliseconds: Double,
)

internal fun repositoryRoot(): Path {
    System.getenv("GEM16_REPO_ROOT")
        ?.takeIf(String::isNotBlank)
        ?.let { return Path.of(it).toAbsolutePath().normalize() }

    var candidate: Path? = Paths.get(System.getProperty("user.dir")).toAbsolutePath().normalize()
    while (candidate != null) {
        if (Files.isRegularFile(candidate.resolve("CMakeLists.txt")) &&
            Files.isDirectory(candidate.resolve("studioApp"))
        ) {
            return candidate
        }
        candidate = candidate.parent
    }
    return Paths.get(System.getProperty("user.dir")).toAbsolutePath().normalize()
}

internal fun defaultServerExecutable(): String {
    val windows = System.getProperty("os.name").contains("Windows", ignoreCase = true)
    val fileName = if (windows) "gem16-server.exe" else "gem16-server"
    System.getProperty("compose.application.resources.dir")
        ?.takeIf(String::isNotBlank)
        ?.let { Path.of(it).resolve("bin").resolve(fileName).normalize() }
        ?.takeIf { Files.isRegularFile(it) }
        ?.let { return it.toString() }
    val relative = if (windows) {
        "build/Windows/blackwell-release/bin/gem16-server.exe"
    } else {
        "build/Linux/blackwell-release/bin/gem16-server"
    }
    return repositoryRoot().resolve(relative).normalize().toString()
}

internal fun defaultModelDirectory(): String =
    HuggingFaceCachePaths.targetView().toAbsolutePath().normalize().toString()

internal fun defaultAssistantDirectory(): String =
    HuggingFaceCachePaths.snapshot(
        Gem16ModelCatalog.assistantRepository,
        Gem16ModelCatalog.assistantRevision,
    ).toAbsolutePath().normalize().toString()

fun String.asAbsolutePath(): String = Path.of(this).toAbsolutePath().normalize().toString()
