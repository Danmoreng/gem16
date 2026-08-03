package com.gem16.studio

import com.gem16.studio.model.StudioSettings
import com.gem16.studio.service.SettingsStore
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class SettingsStoreTest {
    @Test
    fun persistsEditableSystemPromptAndLocalToolPreference() {
        val directory = Files.createTempDirectory("gem16-settings-test")
        try {
            val store = SettingsStore(directory.resolve("settings.properties"))
            val expected = StudioSettings().copy(
                generation = StudioSettings().generation.copy(
                    systemPrompt = "Answer concisely.\nUse German.",
                    localDateTimeTools = false,
                ),
            )
            store.save(expected)
            val loaded = store.load()
            assertEquals(expected.generation.systemPrompt, loaded.generation.systemPrompt)
            assertFalse(loaded.generation.localDateTimeTools)
        } finally {
            directory.toFile().deleteRecursively()
        }
    }

    @Test
    fun developmentServerOverrideDoesNotReplacePersistedInstallerPath() {
        val directory = Files.createTempDirectory("gem16-settings-override-test")
        try {
            val settingsFile = directory.resolve("settings.properties")
            val installerServer = directory.resolve("installed-server.exe").toString()
            val workspaceServer = directory.resolve("workspace-server.exe").toString()
            val regularStore = SettingsStore(settingsFile, serverExecutableOverride = null)
            regularStore.save(
                StudioSettings().copy(
                    server = StudioSettings().server.copy(executable = installerServer),
                ),
            )

            val developmentStore = SettingsStore(
                settingsFile,
                serverExecutableOverride = workspaceServer,
            )
            val developmentSettings = developmentStore.load()
            assertEquals(workspaceServer, developmentSettings.server.executable)
            developmentStore.save(
                developmentSettings.copy(darkTheme = !developmentSettings.darkTheme),
            )

            val persisted = regularStore.load()
            assertEquals(installerServer, persisted.server.executable)
            assertTrue(persisted.darkTheme != StudioSettings().darkTheme)
        } finally {
            directory.toFile().deleteRecursively()
        }
    }
}
