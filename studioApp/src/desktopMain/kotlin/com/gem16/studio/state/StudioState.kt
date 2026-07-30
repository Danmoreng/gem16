package com.gem16.studio.state

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import com.gem16.studio.model.ChatMessage
import com.gem16.studio.model.GenerationConfig
import com.gem16.studio.model.ServerConfig
import com.gem16.studio.model.ServerPhase
import com.gem16.studio.model.StudioSettings
import com.gem16.studio.model.Usage
import com.gem16.studio.service.ChatDelta
import com.gem16.studio.service.Gem16ApiClient
import com.gem16.studio.service.ServerManager
import com.gem16.studio.service.SettingsStore
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class StudioState(
    private val settingsStore: SettingsStore = SettingsStore(),
    val serverManager: ServerManager = ServerManager(),
    private val api: Gem16ApiClient = Gem16ApiClient(),
) : AutoCloseable {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    var settings by mutableStateOf(settingsStore.load())
        private set
    val messages = mutableStateListOf<ChatMessage>()
    var draft by mutableStateOf("")
    var isGenerating by mutableStateOf(false)
        private set
    var chatError by mutableStateOf<String?>(null)
        private set
    var usage by mutableStateOf<Usage?>(null)
        private set
    var sessionId by mutableStateOf<String?>(null)
        private set
    var lastFinishReason by mutableStateOf<String?>(null)
        private set

    private var generationJob: Job? = null

    init {
        serverManager.configure(settings.server)
        if (settings.server.autoStart) {
            scope.launch {
                delay(2_000)
                if (serverManager.phase.value == ServerPhase.Stopped) startServer()
            }
        }
    }

    fun updateServer(transform: (ServerConfig) -> ServerConfig) {
        settings = settings.copy(server = transform(settings.server))
        serverManager.configure(settings.server)
        persist()
    }

    fun updateGeneration(transform: (GenerationConfig) -> GenerationConfig) {
        settings = settings.copy(generation = transform(settings.generation))
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

    fun sendMessage() {
        val text = draft.trim()
        if (text.isEmpty() || isGenerating) return
        if (serverManager.health.value == null) {
            chatError = "The gem16 server is not reachable. Start it on the Server screen first."
            return
        }
        chatError = null
        usage = null
        lastFinishReason = null
        val user = ChatMessage(role = "user", content = text)
        messages += user
        draft = ""
        val requestHistory = messages.toList()
        val assistant = ChatMessage(role = "assistant", content = "", streaming = true)
        messages += assistant
        isGenerating = true

        generationJob = scope.launch {
            try {
                val result = api.streamChat(
                    server = settings.server,
                    generation = settings.generation,
                    messages = requestHistory,
                    sessionId = sessionId,
                ) { delta ->
                    withContext(Dispatchers.Main.immediate) {
                        updateAssistant(assistant.id, delta)
                    }
                }
                sessionId = result.sessionId
                usage = result.usage
                lastFinishReason = result.finishReason
                replaceMessage(assistant.id) { it.copy(streaming = false) }
            } catch (_: CancellationException) {
                replaceMessage(assistant.id) {
                    it.copy(streaming = false, error = "Generation cancelled")
                }
            } catch (error: Exception) {
                val message = error.message ?: "Generation failed"
                chatError = message
                replaceMessage(assistant.id) { it.copy(streaming = false, error = message) }
                if (message.contains("404") || message.contains("unknown", ignoreCase = true)) {
                    sessionId = null
                }
            } finally {
                isGenerating = false
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
        sessionId = null
        usage = null
        lastFinishReason = null
        chatError = null
    }

    fun removeLastExchange() {
        if (isGenerating || messages.isEmpty()) return
        if (messages.lastOrNull()?.role == "assistant") messages.removeLastOrNull()
        if (messages.lastOrNull()?.role == "user") messages.removeLastOrNull()
        // A resident server cannot roll back K/V. Start a new root from the
        // remaining visible history on the next request.
        sessionId = null
    }

    private fun updateAssistant(id: String, delta: ChatDelta) {
        when (delta) {
            is ChatDelta.Text -> replaceMessage(id) { it.copy(content = it.content + delta.value) }
            is ChatDelta.Reasoning -> replaceMessage(id) { it.copy(reasoning = it.reasoning + delta.value) }
            is ChatDelta.Finished -> {
                usage = delta.usage ?: usage
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

    override fun close() {
        api.cancelActive()
        generationJob?.cancel()
        serverManager.close()
        scope.cancel()
    }
}
