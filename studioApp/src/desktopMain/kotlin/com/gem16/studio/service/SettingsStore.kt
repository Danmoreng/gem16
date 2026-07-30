package com.gem16.studio.service

import com.gem16.studio.model.GenerationConfig
import com.gem16.studio.model.ServerConfig
import com.gem16.studio.model.StudioSettings
import com.gem16.studio.model.ThinkingEffort
import com.gem16.studio.model.repositoryRoot
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.util.Properties

class SettingsStore(
    private val settingsFile: Path = Path.of(
        System.getProperty("user.home"),
        ".gem16-studio",
        "settings.properties",
    ),
) {
    fun load(): StudioSettings {
        if (!Files.isRegularFile(settingsFile)) return StudioSettings()
        return runCatching {
            val properties = Properties()
            Files.newInputStream(settingsFile).use(properties::load)
            val defaults = StudioSettings()
            StudioSettings(
                server = ServerConfig(
                    executable = properties.path("server.executable", defaults.server.executable),
                    modelDirectory = properties.path("server.model", defaults.server.modelDirectory),
                    assistantModelDirectory = properties.path(
                        "server.assistant",
                        defaults.server.assistantModelDirectory,
                    ),
                    modelName = properties.text("server.modelName", defaults.server.modelName),
                    host = properties.text("server.host", defaults.server.host),
                    port = properties.int("server.port", defaults.server.port, 1..65535),
                    maxContextTokens = properties.long(
                        "server.maxContext",
                        defaults.server.maxContextTokens,
                        1L..262144L,
                    ),
                    maxSessions = properties.int("server.maxSessions", defaults.server.maxSessions, 1..64),
                    mtpDraftTokens = properties.int("server.mtpDraftTokens", defaults.server.mtpDraftTokens)
                        .takeIf { it == 0 || it == 1 || it == 2 || it == 4 }
                        ?: defaults.server.mtpDraftTokens,
                    mtpAdaptive = properties.bool("server.mtpAdaptive", defaults.server.mtpAdaptive),
                    greedy = properties.bool("server.greedy", defaults.server.greedy),
                ),
                generation = GenerationConfig(
                    thinking = properties.getProperty("generation.thinking")
                        ?.let { value -> ThinkingEffort.entries.firstOrNull { it.wireValue == value } }
                        ?: defaults.generation.thinking,
                    maxOutputTokens = properties.long(
                        "generation.maxOutput",
                        defaults.generation.maxOutputTokens,
                        1L..262144L,
                    ),
                    showReasoning = properties.bool(
                        "generation.showReasoning",
                        defaults.generation.showReasoning,
                    ),
                ),
                darkTheme = properties.bool("ui.darkTheme", defaults.darkTheme),
            )
        }.getOrElse { StudioSettings() }
    }

    fun save(settings: StudioSettings) {
        Files.createDirectories(settingsFile.parent)
        val properties = Properties().apply {
            setProperty("server.executable", settings.server.executable)
            setProperty("server.model", settings.server.modelDirectory)
            setProperty("server.assistant", settings.server.assistantModelDirectory)
            setProperty("server.modelName", settings.server.modelName)
            setProperty("server.host", settings.server.host)
            setProperty("server.port", settings.server.port.toString())
            setProperty("server.maxContext", settings.server.maxContextTokens.toString())
            setProperty("server.maxSessions", settings.server.maxSessions.toString())
            setProperty("server.mtpDraftTokens", settings.server.mtpDraftTokens.toString())
            setProperty("server.mtpAdaptive", settings.server.mtpAdaptive.toString())
            setProperty("server.greedy", settings.server.greedy.toString())
            setProperty("generation.thinking", settings.generation.thinking.wireValue)
            setProperty("generation.maxOutput", settings.generation.maxOutputTokens.toString())
            setProperty("generation.showReasoning", settings.generation.showReasoning.toString())
            setProperty("ui.darkTheme", settings.darkTheme.toString())
        }
        val temporary = settingsFile.resolveSibling("${settingsFile.fileName}.tmp")
        Files.newOutputStream(temporary).use { properties.store(it, "gem16 Studio settings") }
        runCatching {
            Files.move(
                temporary,
                settingsFile,
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.ATOMIC_MOVE,
            )
        }.getOrElse {
            Files.move(temporary, settingsFile, StandardCopyOption.REPLACE_EXISTING)
        }
    }
}

private fun Properties.text(key: String, fallback: String): String =
    getProperty(key)?.trim()?.takeIf(String::isNotEmpty) ?: fallback

private fun Properties.path(key: String, fallback: String): String {
    val value = text(key, fallback)
    val valuePath = runCatching { Path.of(value).toAbsolutePath().normalize() }.getOrNull() ?: return value
    val fallbackPath = Path.of(fallback).toAbsolutePath().normalize()
    val relativeDefault = runCatching { repositoryRoot().relativize(fallbackPath) }.getOrNull()
        ?: return value
    val legacyPath = repositoryRoot().resolve("studioApp").resolve(relativeDefault).normalize()
    return if (valuePath == legacyPath && Files.exists(fallbackPath)) fallback else value
}

private fun Properties.bool(key: String, fallback: Boolean): Boolean =
    getProperty(key)?.trim()?.lowercase()?.let {
        when (it) {
            "true" -> true
            "false" -> false
            else -> null
        }
    } ?: fallback

private fun Properties.int(key: String, fallback: Int, range: IntRange? = null): Int =
    getProperty(key)?.toIntOrNull()?.takeIf { range == null || it in range } ?: fallback

private fun Properties.long(key: String, fallback: Long, range: LongRange): Long =
    getProperty(key)?.toLongOrNull()?.takeIf { it in range } ?: fallback
