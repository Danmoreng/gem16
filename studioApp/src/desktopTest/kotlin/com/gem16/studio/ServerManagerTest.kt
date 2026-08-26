package com.gem16.studio

import com.gem16.studio.model.GenerationConfig
import com.gem16.studio.model.Gem16ModelCatalog
import com.gem16.studio.model.HuggingFaceCachePaths
import com.gem16.studio.model.ServerConfig
import com.gem16.studio.model.ModelProfile
import com.gem16.studio.model.StudioSettings
import com.gem16.studio.model.ThinkingEffort
import com.gem16.studio.model.repositoryRoot
import com.gem16.studio.model.ServerPhase
import com.gem16.studio.service.ServerManager
import com.gem16.studio.service.SettingsStore
import com.gem16.studio.service.buildServerCommand
import com.sun.net.httpserver.HttpServer
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import java.net.InetSocketAddress
import java.nio.file.Files
import java.util.concurrent.Executors
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ServerManagerTest {
    @Test
    fun startAttachesBeforeValidatingLocalPaths() {
        runBlocking {
            val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
            server.executor = Executors.newSingleThreadExecutor()
            server.createContext("/health") { exchange ->
                val body = """{"status":"ok","resident_sessions":0,"session_limit":1,"max_context_tokens":32768,"mtp_draft_tokens":2,"sampling":{"enabled":true}}"""
                    .toByteArray()
                exchange.sendResponseHeaders(200, body.size.toLong())
                exchange.responseBody.use { it.write(body) }
            }
            server.start()
            val manager = ServerManager()
            try {
                manager.start(
                    ServerConfig(
                        executable = "/missing/gem16-server",
                        modelDirectory = "/missing/model",
                        assistantModelDirectory = "/missing/assistant",
                        port = server.address.port,
                    ),
                )
                withTimeout(5_000) { manager.phase.first { it == ServerPhase.External } }
                assertEquals("ok", manager.health.value?.status)
                assertEquals(null, manager.error.value)
            } finally {
                manager.close()
                server.stop(0)
                (server.executor as java.util.concurrent.ExecutorService).shutdownNow()
            }
        }
    }

    @Test
    fun defaultsResolveRepositoryExecutableAndHuggingFaceModels() {
        val config = ServerConfig()
        val root = repositoryRoot()
        val executable = if (System.getProperty("os.name").contains("Windows", ignoreCase = true)) {
            "build/Windows/blackwell-release/bin/gem16-server.exe"
        } else {
            "build/Linux/blackwell-release/bin/gem16-server"
        }
        assertEquals(root.resolve(executable).normalize().toString(), config.executable)
        assertEquals(
            HuggingFaceCachePaths.targetView().toAbsolutePath().normalize().toString(),
            config.modelDirectory,
        )
        assertEquals(
            HuggingFaceCachePaths.snapshot(
                Gem16ModelCatalog.assistantRepository,
                Gem16ModelCatalog.assistantRevision,
            ).toAbsolutePath().normalize().toString(),
            config.assistantModelDirectory,
        )
    }

    @Test
    fun commandIncludesExplicitRuntimeConfiguration() {
        val command = buildServerCommand(
            ServerConfig(
                executable = "/opt/gem16-server",
                modelDirectory = "/models/target",
                assistantModelDirectory = "/models/assistant",
                modelName = "local-gemma",
                host = "127.0.0.1",
                port = 9090,
                maxContextTokens = 65536,
                maxSessions = 2,
                mtpDraftTokens = 2,
                mtpAdaptive = true,
            ),
        )
        assertEquals("/opt/gem16-server", command.first())
        assertTrue(command.containsAll(listOf("--model", "/models/target", "--assistant-model", "/models/assistant")))
        assertTrue(command.containsAll(listOf("--mtp-draft-tokens", "2", "--mtp-adaptive")))
        assertFalse(command.contains("--greedy"))
    }

    @Test
    fun commandOmitsAssistantWhenMtpIsOff() {
        val command = buildServerCommand(
            ServerConfig(
                executable = "server",
                modelDirectory = "model",
                mtpDraftTokens = 0,
                greedy = true,
            ),
        )
        assertTrue(command.contains("--greedy"))
        assertFalse(command.contains("--assistant-model"))
        assertFalse(command.contains("--mtp-draft-tokens"))
    }

    @Test
    fun gemma26BProfileUsesTextOnlySingleSlotFixedMtpContract() {
        val config = ServerConfig().selectProfile(ModelProfile.Gemma4Moe26BA4B).copy(
            executable = "server",
            modelDirectory = "/models/26b-target",
            assistantModelDirectory = "/models/26b-assistant",
        )
        assertEquals(ModelProfile.Gemma4Moe26BA4B, config.modelProfile)
        assertFalse(config.supportsMedia)
        assertEquals(1, config.maxSessions)
        assertEquals(2, config.mtpDraftTokens)
        assertFalse(config.mtpAdaptive)
        val command = buildServerCommand(config)
        assertTrue(command.containsAll(listOf("--model", "/models/26b-target")))
        assertTrue(command.containsAll(listOf("--assistant-model", "/models/26b-assistant")))
        assertTrue(command.containsAll(listOf("--mtp-draft-tokens", "2")))
        assertFalse(command.contains("--mtp-adaptive"))
    }

    @Test
    fun settingsRoundTrip() {
        val directory = Files.createTempDirectory("gem16-studio-test")
        val store = SettingsStore(directory.resolve("settings.properties"))
        val expected = StudioSettings(
            server = ServerConfig(
                modelProfile = ModelProfile.Gemma4Moe26BA4B,
                executable = "server.exe",
                modelDirectory = "target",
                assistantModelDirectory = "assistant",
                modelName = "gemma",
                port = 7777,
                maxContextTokens = 131072,
                maxSessions = 3,
                mtpDraftTokens = 4,
                mtpAdaptive = true,
                greedy = true,
            ),
            generation = GenerationConfig(
                thinking = ThinkingEffort.High,
                maxOutputTokens = 16384,
                showReasoning = false,
            ),
            darkTheme = false,
        )
        store.save(expected)
        assertEquals(expected, store.load())
        directory.toFile().deleteRecursively()
    }
}
