package com.gem16.studio.state

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.gem16.studio.model.ChatMessage
import com.gem16.studio.model.ChatActivity
import com.gem16.studio.model.ChatActivityPhase
import com.gem16.studio.model.GenerationConfig
import com.gem16.studio.model.MediaAttachment
import com.gem16.studio.model.MediaKind
import com.gem16.studio.model.PerformanceStats
import com.gem16.studio.model.ServerConfig
import com.gem16.studio.model.StudioSettings
import com.gem16.studio.model.StreamPerformanceStats
import com.gem16.studio.model.Usage
import com.gem16.studio.service.AudioRecorder
import com.gem16.studio.service.ChatDelta
import com.gem16.studio.service.ChatRequestPhase
import com.gem16.studio.service.Gem16ApiClient
import com.gem16.studio.service.LocalToolExecutor
import com.gem16.studio.service.MaxEncodedMediaBytes
import com.gem16.studio.service.MaxLocalToolRounds
import com.gem16.studio.service.ModelManager
import com.gem16.studio.service.ServerManager
import com.gem16.studio.service.SettingsStore
import com.gem16.studio.service.clipboardContainsImage
import com.gem16.studio.service.encodedMediaBytes
import com.gem16.studio.service.formatBytes
import com.gem16.studio.service.loadClipboardImageAttachment
import com.gem16.studio.service.loadMediaAttachment
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.nio.file.Files
import java.nio.file.Path

class StudioState(
    private val settingsStore: SettingsStore = SettingsStore(),
    val serverManager: ServerManager = ServerManager(),
    val modelManager: ModelManager = ModelManager(),
    private val api: Gem16ApiClient = Gem16ApiClient(),
    private val audioRecorder: AudioRecorder = AudioRecorder(),
    private val localTools: LocalToolExecutor = LocalToolExecutor(),
) : AutoCloseable {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    var settings by mutableStateOf(settingsStore.load())
        private set
    val messages = mutableStateListOf<ChatMessage>()
    val pendingAttachments = mutableStateListOf<MediaAttachment>()
    var draft by mutableStateOf("")
    var isGenerating by mutableStateOf(false)
        private set
    var isLoadingAttachments by mutableStateOf(false)
        private set
    var isRecording by mutableStateOf(false)
        private set
    var recordingMillis by mutableStateOf(0L)
        private set
    var recordingLevel by mutableStateOf(0f)
        private set
    var chatError by mutableStateOf<String?>(null)
        private set
    var chatActivity by mutableStateOf<ChatActivity?>(null)
        private set
    var usage by mutableStateOf<Usage?>(null)
        private set
    var contextTokensUsed by mutableStateOf(0L)
        private set
    var performance by mutableStateOf<PerformanceStats?>(null)
        private set
    var livePerformance by mutableStateOf<StreamPerformanceStats?>(null)
        private set
    var sessionId by mutableStateOf<String?>(null)
        private set
    var lastFinishReason by mutableStateOf<String?>(null)
        private set

    private var generationJob: Job? = null
    private var recordingJob: Job? = null
    private var modelDownloadJob: Job? = null
    private var discardRecording = false

    init {
        serverManager.configure(settings.server)
        if (serverConfigurationExists()) startServer()
    }

    fun updateServer(transform: (ServerConfig) -> ServerConfig) {
        settings = settings.copy(server = transform(settings.server))
        serverManager.configure(settings.server)
        persist()
    }

    fun updateGeneration(transform: (GenerationConfig) -> GenerationConfig) {
        val previous = settings.generation
        val updated = transform(previous)
        settings = settings.copy(generation = updated)
        if (
            previous.systemPrompt != updated.systemPrompt ||
            previous.localDateTimeTools != updated.localDateTimeTools
        ) {
            sessionId = null
            contextTokensUsed = 0L
        }
        persist()
    }

    fun toggleTheme() {
        settings = settings.copy(darkTheme = !settings.darkTheme)
        persist()
    }

    fun startServer() {
        sessionId = null
        serverManager.start(settings.server)
    }

    fun stopServer() {
        cancelGeneration()
        sessionId = null
        serverManager.stop()
    }

    fun downloadModels(token: String?) {
        if (modelDownloadJob?.isActive == true) return
        modelDownloadJob = scope.launch {
            try {
                val installed = modelManager.downloadAll(token)
                updateServer {
                    it.copy(
                        modelDirectory = installed.targetDirectory.toString(),
                        assistantModelDirectory = installed.assistantDirectory.toString(),
                    )
                }
            } catch (_: Exception) {
                // ModelManager exposes a user-facing error in its StateFlow.
            } finally {
                modelDownloadJob = null
            }
        }
    }

    fun cancelModelDownload() {
        modelManager.cancel()
    }

    fun useCachedModels() {
        modelManager.refresh()
        val installed = modelManager.state.value
        if (!installed.allReady) return
        updateServer {
            it.copy(
                modelDirectory = installed.targetDirectory.toString(),
                assistantModelDirectory = installed.assistantDirectory.toString(),
            )
        }
    }

    fun startRecording() {
        if (isGenerating || isLoadingAttachments || isRecording) return
        chatError = null
        recordingMillis = 0L
        recordingLevel = 0f
        discardRecording = false
        isRecording = true
        recordingJob = scope.launch {
            try {
                val recording = withContext(Dispatchers.IO) {
                    audioRecorder.record { durationMillis, level ->
                        scope.launch {
                            recordingMillis = durationMillis
                            recordingLevel = level
                        }
                    }
                }
                if (!discardRecording) {
                    val attachment = MediaAttachment(
                        fileName = recording.fileName,
                        kind = MediaKind.Audio,
                        mimeType = "audio/wav",
                        format = "wav",
                        bytes = recording.wavBytes,
                        durationMillis = recording.durationMillis,
                    )
                    val encoded = encodedMediaBytes(messages) +
                        pendingAttachments.sumOf(MediaAttachment::encodedSize) + attachment.encodedSize
                    if (encoded <= MaxEncodedMediaBytes) {
                        pendingAttachments += attachment
                    } else {
                        chatError = "The recording would exceed gem16's ${formatBytes(MaxEncodedMediaBytes)} " +
                            "encoded-media request limit. Start a new chat first."
                    }
                }
            } catch (error: Exception) {
                if (!discardRecording) chatError = error.message ?: "Microphone recording failed"
            } finally {
                isRecording = false
                recordingMillis = 0L
                recordingLevel = 0f
                recordingJob = null
            }
        }
    }

    fun stopRecording() {
        if (isRecording) audioRecorder.stop()
    }

    fun cancelRecording() {
        if (!isRecording) return
        discardRecording = true
        audioRecorder.stop()
    }

    fun addAttachments(paths: List<Path>) {
        if (paths.isEmpty() || isGenerating || isLoadingAttachments || isRecording) return
        isLoadingAttachments = true
        val pdfCount = paths.count { it.fileName.toString().endsWith(".pdf", ignoreCase = true) }
        chatActivity = ChatActivity(
            phase = ChatActivityPhase.ProcessingAttachment,
            detail = if (pdfCount > 0) {
                "Extracting text locally from ${if (pdfCount == 1) "PDF" else "$pdfCount PDFs"}"
            } else {
                "Reading ${if (paths.size == 1) "attachment" else "${paths.size} attachments"} locally"
            },
        )
        scope.launch {
            try {
                val loaded = withContext(Dispatchers.IO) { paths.map(::loadMediaAttachment) }
                val failures = loaded.mapNotNull { it.exceptionOrNull() }
                var encoded = encodedMediaBytes(messages) + pendingAttachments.sumOf(MediaAttachment::encodedSize)
                loaded.mapNotNull { it.getOrNull() }.forEach { attachment ->
                    if (encoded + attachment.encodedSize <= MaxEncodedMediaBytes) {
                        pendingAttachments += attachment
                        encoded += attachment.encodedSize
                    } else {
                        chatError = "Attachments would exceed gem16's ${formatBytes(MaxEncodedMediaBytes)} " +
                            "encoded-media request limit. Start a new chat or remove an attachment."
                    }
                }
                if (failures.isNotEmpty()) {
                    chatError = failures.joinToString("\n") { it.message ?: "Could not load media file" }
                }
            } finally {
                isLoadingAttachments = false
                if (chatActivity?.phase == ChatActivityPhase.ProcessingAttachment) chatActivity = null
            }
        }
    }

    fun addClipboardImage(): Boolean {
        if (isGenerating || isLoadingAttachments || isRecording || !clipboardContainsImage()) return false
        isLoadingAttachments = true
        chatError = null
        chatActivity = ChatActivity(ChatActivityPhase.ProcessingAttachment, "Reading clipboard image locally")
        scope.launch {
            try {
                val result = withContext(Dispatchers.IO) { loadClipboardImageAttachment() }
                result.onSuccess { attachment ->
                    val encoded = encodedMediaBytes(messages) +
                        pendingAttachments.sumOf(MediaAttachment::encodedSize)
                    if (encoded + attachment.encodedSize <= MaxEncodedMediaBytes) {
                        pendingAttachments += attachment
                    } else {
                        chatError = "Attachments would exceed gem16's ${formatBytes(MaxEncodedMediaBytes)} " +
                            "encoded-media request limit. Start a new chat or remove an attachment."
                    }
                }.onFailure { error ->
                    chatError = error.message ?: "Could not read the clipboard image"
                }
            } finally {
                isLoadingAttachments = false
                if (chatActivity?.phase == ChatActivityPhase.ProcessingAttachment) chatActivity = null
            }
        }
        return true
    }

    fun removeAttachment(id: String) {
        if (!isGenerating && !isRecording) pendingAttachments.removeAll { it.id == id }
    }

    fun sendMessage() {
        val text = draft.trim()
        val attachments = pendingAttachments.toList()
        if ((text.isEmpty() && attachments.isEmpty()) || isGenerating || isLoadingAttachments || isRecording) return
        if (serverManager.health.value == null) {
            chatError = "The gem16 server is not reachable. Start it on the Server screen first."
            return
        }
        chatError = null
        usage = null
        performance = null
        livePerformance = null
        lastFinishReason = null
        val proposedHistory = messages + ChatMessage(role = "user", content = text, attachments = attachments)
        val encodedMedia = encodedMediaBytes(proposedHistory)
        if (encodedMedia > MaxEncodedMediaBytes) {
            chatError = "This conversation contains ${formatBytes(encodedMedia)} of encoded media; " +
                "the gem16 limit is ${formatBytes(MaxEncodedMediaBytes)}. Start a new chat."
            return
        }
        val user = proposedHistory.last()
        messages += user
        draft = ""
        pendingAttachments.clear()
        var requestHistory = messages.toList()
        val assistant = ChatMessage(role = "assistant", content = "", streaming = true)
        messages += assistant
        var currentAssistantId = assistant.id
        isGenerating = true
        chatActivity = ChatActivity(ChatActivityPhase.PreparingRequest, "Preparing request")

        generationJob = scope.launch {
            try {
                var toolRounds = 0
                var totalPerformance: PerformanceStats? = null
                while (true) {
                    val response = api.streamChat(
                        server = settings.server,
                        generation = settings.generation,
                        messages = requestHistory,
                        sessionId = sessionId,
                        onPhase = { phase ->
                            withContext(Dispatchers.Main.immediate) {
                                updateRequestActivity(phase)
                            }
                        },
                        onProgress = { progress ->
                            withContext(Dispatchers.Main.immediate) { livePerformance = progress }
                        },
                        onDelta = { delta ->
                            withContext(Dispatchers.Main.immediate) {
                                updateAssistant(currentAssistantId, delta)
                            }
                        },
                    )
                    sessionId = response.sessionId
                    response.usage?.let { measured ->
                        usage = measured
                        contextTokensUsed = measured.totalTokens
                    }
                    totalPerformance = combinePerformance(totalPerformance, response.performance)
                    if (response.toolCalls.isEmpty()) {
                        val completed = messages.firstOrNull { it.id == currentAssistantId }
                        check(completed != null && completed.content.isNotBlank()) {
                            if (completed?.reasoning?.isNotBlank() == true) {
                                "Generation ended after reasoning without a visible answer " +
                                    "(finish reason: ${response.finishReason ?: "missing"})."
                            } else {
                                "The server completed the response without returning text, reasoning, or a tool call."
                            }
                        }
                        performance = totalPerformance
                        lastFinishReason = response.finishReason
                        replaceMessage(currentAssistantId) { it.copy(streaming = false) }
                        break
                    }
                    toolRounds += 1
                    check(toolRounds <= MaxLocalToolRounds) {
                        "Gemma exceeded the limit of $MaxLocalToolRounds consecutive local tool rounds."
                    }
                    updateActivity(
                        ChatActivityPhase.RunningTool,
                        "Running ${response.toolCalls.joinToString { localToolLabel(it.name) }} locally",
                    )
                    val completedCalls = response.toolCalls.map { call ->
                        call.copy(resultJson = localTools.execute(call))
                    }
                    replaceMessage(currentAssistantId) {
                        it.copy(streaming = false, toolCalls = completedCalls)
                    }
                    completedCalls.forEach { call ->
                        messages += ChatMessage(
                            role = "tool",
                            content = requireNotNull(call.resultJson),
                            toolCallId = call.id,
                        )
                    }
                    val nextAssistant = ChatMessage(role = "assistant", content = "", streaming = true)
                    messages += nextAssistant
                    currentAssistantId = nextAssistant.id
                    requestHistory = messages.filterNot { it.id == currentAssistantId }
                    livePerformance = null
                }
            } catch (_: CancellationException) {
                replaceMessage(currentAssistantId) {
                    it.copy(streaming = false, error = "Generation cancelled")
                }
            } catch (error: Exception) {
                val message = error.message ?: "Generation failed"
                chatError = message
                replaceMessage(currentAssistantId) { it.copy(streaming = false, error = message) }
                if (message.contains("404") || message.contains("unknown", ignoreCase = true)) {
                    sessionId = null
                }
            } finally {
                isGenerating = false
                chatActivity = null
                livePerformance = null
                generationJob = null
            }
        }
    }

    fun cancelGeneration() {
        api.cancelActive()
        generationJob?.cancel()
    }

    fun clearChat() {
        if (isGenerating) return
        messages.clear()
        pendingAttachments.clear()
        sessionId = null
        usage = null
        contextTokensUsed = 0L
        performance = null
        livePerformance = null
        lastFinishReason = null
        chatError = null
    }

    fun removeLastExchange() {
        if (isGenerating || messages.isEmpty()) return
        val userIndex = messages.indexOfLast { it.role == "user" }
        if (userIndex < 0) return
        while (messages.size > userIndex) messages.removeAt(messages.lastIndex)
        // A resident server cannot roll back K/V. Start a new root from the
        // remaining visible history on the next request.
        sessionId = null
        contextTokensUsed = 0L
    }

    private fun updateAssistant(id: String, delta: ChatDelta) {
        when (delta) {
            is ChatDelta.Text -> replaceMessage(id) { it.copy(content = it.content + delta.value) }
            is ChatDelta.Reasoning -> replaceMessage(id) { it.copy(reasoning = it.reasoning + delta.value) }
            is ChatDelta.Finished -> {
                delta.usage?.let { measured ->
                    usage = measured
                    contextTokensUsed = measured.totalTokens
                }
                lastFinishReason = delta.reason ?: lastFinishReason
            }
        }
    }

    private fun replaceMessage(id: String, transform: (ChatMessage) -> ChatMessage) {
        val index = messages.indexOfFirst { it.id == id }
        if (index >= 0) messages[index] = transform(messages[index])
    }

    private fun persist() {
        runCatching { settingsStore.save(settings) }
            .onFailure { chatError = "Could not save settings: ${it.message}" }
    }

    private fun updateRequestActivity(phase: ChatRequestPhase) {
        when (phase) {
            ChatRequestPhase.PreparingRequest -> updateActivity(
                ChatActivityPhase.PreparingRequest,
                "Preparing and sending request",
            )
            ChatRequestPhase.WaitingForFirstToken -> updateActivity(
                ChatActivityPhase.WaitingForFirstToken,
                "Waiting for first token · the server may be prefilling",
            )
            ChatRequestPhase.Decoding -> updateActivity(
                ChatActivityPhase.Decoding,
                "Decoding response",
            )
        }
    }

    private fun updateActivity(phase: ChatActivityPhase, detail: String) {
        chatActivity = ChatActivity(
            phase = phase,
            detail = detail,
            startedNanos = chatActivity?.startedNanos ?: System.nanoTime(),
        )
    }

    private fun serverConfigurationExists(): Boolean {
        return runCatching {
            val server = settings.server
            Files.isRegularFile(Path.of(server.executable)) &&
                Files.isDirectory(Path.of(server.modelDirectory)) &&
                (server.mtpDraftTokens == 0 || Files.isDirectory(Path.of(server.assistantModelDirectory)))
        }.getOrDefault(false)
    }

    override fun close() {
        api.cancelActive()
        generationJob?.cancel()
        discardRecording = true
        audioRecorder.close()
        recordingJob?.cancel()
        modelManager.cancel()
        modelDownloadJob?.cancel()
        serverManager.close()
        scope.cancel()
    }
}

private fun combinePerformance(
    first: PerformanceStats?,
    second: PerformanceStats?,
): PerformanceStats? {
    if (first == null) return second
    if (second == null) return first
    val prefillMilliseconds = first.prefillMilliseconds + second.prefillMilliseconds
    val decodeMilliseconds = first.decodeMilliseconds + second.decodeMilliseconds
    val prefillTokens = first.prefillTokensPerSecond * first.prefillMilliseconds / 1_000.0 +
        second.prefillTokensPerSecond * second.prefillMilliseconds / 1_000.0
    val decodeTokens = first.decodeTokensPerSecond * first.decodeMilliseconds / 1_000.0 +
        second.decodeTokensPerSecond * second.decodeMilliseconds / 1_000.0
    return PerformanceStats(
        decodeTokensPerSecond = if (decodeMilliseconds > 0.0) decodeTokens * 1_000.0 / decodeMilliseconds else 0.0,
        prefillTokensPerSecond = if (prefillMilliseconds > 0.0) {
            prefillTokens * 1_000.0 / prefillMilliseconds
        } else {
            0.0
        },
        prefillMilliseconds = prefillMilliseconds,
        decodeMilliseconds = decodeMilliseconds,
    )
}

private fun localToolLabel(name: String): String = when (name) {
    "get_current_date" -> "current-date tool"
    "get_current_time" -> "current-time tool"
    else -> name
}
