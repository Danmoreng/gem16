package com.gem16.studio.service

import com.gem16.studio.model.HealthSnapshot
import com.gem16.studio.model.ServerConfig
import com.gem16.studio.model.ServerPhase
import com.gem16.studio.model.repositoryRoot
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import java.net.URI
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.time.Duration
import java.time.Instant
import java.util.concurrent.TimeUnit

class ServerManager : AutoCloseable {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val json = Json { ignoreUnknownKeys = true }
    private val http = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(2))
        .build()

    private val _phase = MutableStateFlow(ServerPhase.Stopped)
    val phase: StateFlow<ServerPhase> = _phase.asStateFlow()

    private val _health = MutableStateFlow<HealthSnapshot?>(null)
    val health: StateFlow<HealthSnapshot?> = _health.asStateFlow()

    private val _logs = MutableStateFlow<List<String>>(emptyList())
    val logs: StateFlow<List<String>> = _logs.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    @Volatile
    private var process: Process? = null
    @Volatile
    private var config = ServerConfig()
    private var readerJob: Job? = null
    private var waiterJob: Job? = null

    init {
        scope.launch {
            while (isActive) {
                pollHealth()
                delay(1_500)
            }
        }
    }

    fun configure(value: ServerConfig) {
        config = value
    }

    fun start(value: ServerConfig) {
        if (process?.isAlive == true || _phase.value == ServerPhase.Starting) return
        config = value
        _error.value = null
        _logs.value = emptyList()
        _health.value = null
        _phase.value = ServerPhase.Starting
        appendLog("Checking ${value.host}:${value.port} for an existing server")
        scope.launch {
            try {
                val existing = runCatching { fetchHealth(value) }.getOrNull()
                if (existing != null) {
                    _health.value = existing
                    _phase.value = ServerPhase.External
                    appendLog("Attached to existing server at ${value.host}:${value.port}")
                    return@launch
                }
                validate(value)?.let { validationError ->
                    _error.value = validationError
                    _phase.value = ServerPhase.Error
                    appendLog("Start failed: $validationError")
                    return@launch
                }
                appendLog("Starting ${value.executable}")
                val command = buildServerCommand(value)
                val started = ProcessBuilder(command)
                    .directory(repositoryRoot().toFile())
                    .redirectErrorStream(true)
                    .start()
                process = started
                readerJob = launch {
                    started.inputStream.bufferedReader().useLines { lines ->
                        lines.forEach(::appendLog)
                    }
                }
                waiterJob = launch {
                    val exitCode = started.waitFor()
                    appendLog("Server exited with code $exitCode")
                    process = null
                    _health.value = null
                    if (_phase.value != ServerPhase.Stopping) {
                        _error.value = if (exitCode == 0) null else "gem16-server exited with code $exitCode"
                        _phase.value = if (exitCode == 0) ServerPhase.Stopped else ServerPhase.Error
                    } else {
                        _phase.value = ServerPhase.Stopped
                    }
                }
            } catch (error: Exception) {
                process = null
                _error.value = error.message ?: "Failed to start gem16-server"
                appendLog("Start failed: ${_error.value}")
                _phase.value = ServerPhase.Error
            }
        }
    }

    fun stop() {
        val active = process
        if (active == null || !active.isAlive) {
            _phase.value = ServerPhase.Stopped
            _health.value = null
            return
        }
        _phase.value = ServerPhase.Stopping
        appendLog("Stopping managed server…")
        scope.launch {
            active.descendants().forEach { child -> runCatching { child.destroy() } }
            active.destroy()
            repeat(30) {
                if (!active.isAlive) return@launch
                delay(100)
            }
            if (active.isAlive) {
                appendLog("Server did not stop in time; terminating it forcibly")
                active.descendants().forEach { child -> runCatching { child.destroyForcibly() } }
                active.destroyForcibly()
            }
        }
    }

    fun clearLogs() {
        _logs.value = emptyList()
    }

    private suspend fun pollHealth() {
        val snapshot = runCatching { fetchHealth(config) }.getOrNull()
        _health.value = snapshot
        val managedAlive = process?.isAlive == true
        if (snapshot != null) {
            _error.value = null
            _phase.value = if (managedAlive) ServerPhase.Running else ServerPhase.External
        } else if (!managedAlive && _phase.value == ServerPhase.External) {
            _phase.value = ServerPhase.Stopped
        }
    }

    private suspend fun fetchHealth(config: ServerConfig): HealthSnapshot = withContext(Dispatchers.IO) {
        val request = HttpRequest.newBuilder(URI.create("http://${config.clientHost}:${config.port}/health"))
            .timeout(Duration.ofSeconds(3))
            .GET()
            .build()
        val response = http.send(request, HttpResponse.BodyHandlers.ofString())
        require(response.statusCode() == 200) { "health returned HTTP ${response.statusCode()}" }
        val root = json.parseToJsonElement(response.body()).jsonObject
        val sampling = root["sampling"]?.jsonObject
        HealthSnapshot(
            status = root["status"]?.jsonPrimitive?.content ?: "unknown",
            residentSessions = root["resident_sessions"]?.jsonPrimitive?.intOrNull ?: 0,
            sessionLimit = root["session_limit"]?.jsonPrimitive?.intOrNull ?: 0,
            maxContextTokens = root["max_context_tokens"]?.jsonPrimitive?.longOrNull ?: 0,
            mtpDraftTokens = root["mtp_draft_tokens"]?.jsonPrimitive?.intOrNull ?: 0,
            samplingEnabled = sampling?.get("enabled")?.jsonPrimitive?.booleanOrNull ?: false,
            temperature = sampling?.get("temperature")?.jsonPrimitive?.doubleOrNull,
            topK = sampling?.get("top_k")?.jsonPrimitive?.intOrNull,
            topP = sampling?.get("top_p")?.jsonPrimitive?.doubleOrNull,
        )
    }

    private fun appendLog(line: String) {
        val stamped = "${Instant.now()}  $line"
        _logs.value = (_logs.value + stamped).takeLast(1_000)
    }

    private fun validate(config: ServerConfig): String? {
        if (!Files.isRegularFile(Path.of(config.executable))) return "Server executable does not exist"
        if (!Files.isDirectory(Path.of(config.modelDirectory))) return "Model directory does not exist"
        if (config.mtpDraftTokens != 0 && !Files.isDirectory(Path.of(config.assistantModelDirectory))) {
            return "MTP is enabled, but the assistant model directory does not exist"
        }
        if (config.port !in 1..65535) return "Port must be in [1, 65535]"
        if (config.maxContextTokens !in 1..262144) return "Context must be in [1, 262144]"
        return null
    }

    override fun close() {
        process?.takeIf(Process::isAlive)?.let { active ->
            active.descendants().forEach { child -> runCatching { child.destroy() } }
            active.destroy()
            if (!runCatching { active.waitFor(3, TimeUnit.SECONDS) }.getOrDefault(false)) {
                active.descendants().forEach { child -> runCatching { child.destroyForcibly() } }
                active.destroyForcibly()
            }
        }
        readerJob?.cancel()
        waiterJob?.cancel()
        scope.cancel()
    }
}

fun buildServerCommand(config: ServerConfig): List<String> = buildList {
    add(config.executable)
    addAll(listOf("--model", config.modelDirectory))
    addAll(listOf("--model-name", config.modelName))
    addAll(listOf("--host", config.host))
    addAll(listOf("--port", config.port.toString()))
    addAll(listOf("--max-context", config.maxContextTokens.toString()))
    addAll(listOf("--max-sessions", config.maxSessions.toString()))
    if (config.greedy) add("--greedy")
    if (config.mtpDraftTokens != 0) {
        addAll(listOf("--assistant-model", config.assistantModelDirectory))
        addAll(listOf("--mtp-draft-tokens", config.mtpDraftTokens.toString()))
        if (config.mtpAdaptive) add("--mtp-adaptive")
    }
}
