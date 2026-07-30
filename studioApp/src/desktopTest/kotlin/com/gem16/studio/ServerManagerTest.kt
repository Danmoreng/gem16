package com.gem16.studio

import com.gem16.studio.model.GenerationConfig
import com.gem16.studio.model.ServerConfig
import com.gem16.studio.model.StudioSettings
import com.gem16.studio.model.ThinkingEffort
import com.gem16.studio.model.repositoryRoot
import com.gem16.studio.service.SettingsStore
import com.gem16.studio.service.buildServerCommand
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ServerManagerTest {
    @Test
    fun defaultsResolveAgainstRepositoryInsteadOfStudioModule() {
        val config = ServerConfig()
        val root = repositoryRoot()
        val executable = if (System.getProperty("os.name").contains("Windows", ignoreCase = true)) {
            "build/Windows/blackwell-release/bin/gem16-server.exe"
        } else {
            "build/Linux/blackwell-release/bin/gem16-server"
        }
        assertEquals(root.resolve(executable).normalize().toString(), config.executable)
        assertEquals(
            root.resolve("models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497")
                .normalize().toString(),
            config.modelDirectory,
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
    fun settingsRoundTrip() {
        val directory = Files.createTempDirectory("gem16-studio-test")
        val store = SettingsStore(directory.resolve("settings.properties"))
        val expected = StudioSettings(
            server = ServerConfig(
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
                autoStart = true,
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
