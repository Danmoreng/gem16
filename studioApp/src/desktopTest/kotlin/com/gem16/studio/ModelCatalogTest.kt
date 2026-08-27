package com.gem16.studio

import com.gem16.studio.model.Gem16ModelCatalog
import com.gem16.studio.model.Gem16Qualified26BModelCatalog
import com.gem16.studio.model.LockedModel
import com.gem16.studio.model.repositoryRoot
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class ModelCatalogTest {
    @Test
    fun `catalog uses immutable revisions and valid hashes`() {
        for (model in listOf(
            Gem16ModelCatalog.target,
            Gem16ModelCatalog.assistant,
            Gem16Qualified26BModelCatalog.target,
            Gem16Qualified26BModelCatalog.assistant,
        )) {
            assertTrue(model.revision.matches(Regex("[0-9a-f]{40}")))
            assertTrue(model.files.isNotEmpty())
            model.files.forEach { file ->
                assertTrue(file.size > 0)
                assertTrue(file.sha256.matches(Regex("[0-9a-f]{64}")))
                assertTrue(file.gitOid.matches(Regex("[0-9a-f]{40}")))
                file.lfsOid?.let { assertTrue(it.matches(Regex("[0-9a-f]{64}"))) }
            }
        }
    }

    @Test
    fun `target view replaces tokenizer config with pinned Google source`() {
        val tokenizer = Gem16ModelCatalog.target.files.single { it.path == "tokenizer_config.json" }
        val source = tokenizer.source(
            Gem16ModelCatalog.target.repository,
            Gem16ModelCatalog.target.revision,
        )

        assertEquals(Gem16ModelCatalog.tokenizerRepository, source.repository)
        assertEquals(Gem16ModelCatalog.tokenizerRevision, source.revision)
        assertEquals("tokenizer_config.json", source.path)
    }

    @Test
    fun `download size is the complete locked payload`() {
        assertEquals(
            Gem16ModelCatalog.target.files.sumOf { it.size } +
                Gem16ModelCatalog.assistant.files.sumOf { it.size },
            Gem16ModelCatalog.totalBytes,
        )
        assertEquals(
            Gem16Qualified26BModelCatalog.target.files.sumOf { it.size } +
                Gem16Qualified26BModelCatalog.assistant.files.sumOf { it.size },
            Gem16Qualified26BModelCatalog.totalBytes,
        )
    }

    @Test
    fun `catalog exactly matches repository lock files`() {
        assertMatchesLock(Gem16ModelCatalog.target, "models/gemma4-12b-nvfp4.lock.json")
        assertMatchesLock(Gem16ModelCatalog.assistant, "models/gemma4-12b-mtp-assistant.lock.json")
        assertMatchesLock(
            Gem16Qualified26BModelCatalog.target,
            "models/gemma4-26b-gem16-target.lock.json",
        )
        assertMatchesLock(
            Gem16Qualified26BModelCatalog.assistant,
            "models/gemma4-26b-gem16-assistant.lock.json",
        )
    }

    private fun assertMatchesLock(model: LockedModel, relativePath: String) {
        val lock = Json.parseToJsonElement(
            repositoryRoot().resolve(relativePath).toFile().readText(),
        ).jsonObject
        assertEquals(model.repository, lock.getValue("repository").jsonPrimitive.content)
        assertEquals(model.revision, lock.getValue("revision").jsonPrimitive.content)

        val lockedFiles = lock.getValue("files").jsonArray.associateBy {
            it.jsonObject.getValue("path").jsonPrimitive.content
        }
        assertEquals(model.files.map { it.path }.toSet(), lockedFiles.keys)
        model.files.forEach { file ->
            val locked = lockedFiles.getValue(file.path).jsonObject
            assertEquals(file.size, locked.getValue("size").jsonPrimitive.content.toLong())
            assertEquals(file.sha256, locked.getValue("sha256").jsonPrimitive.content)
            assertEquals(file.gitOid, locked.getValue("git_oid").jsonPrimitive.content)
            assertEquals(file.lfsOid, locked["lfs_oid"]?.jsonPrimitive?.content)

            val expectedSource = locked["source"]?.jsonObject
            val actualSource = file.source
            assertEquals(expectedSource?.get("repository")?.jsonPrimitive?.content, actualSource?.repository)
            assertEquals(expectedSource?.get("revision")?.jsonPrimitive?.content, actualSource?.revision)
            assertEquals(expectedSource?.get("path")?.jsonPrimitive?.content, actualSource?.path)
        }
    }
}
