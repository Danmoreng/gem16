package com.gem16.studio.service

import com.gem16.studio.model.Gem16ModelCatalog
import com.gem16.studio.model.HuggingFaceCachePaths
import com.gem16.studio.model.HuggingFaceSource
import com.gem16.studio.model.LockedModel
import com.gem16.studio.model.LockedModelFile
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.net.URI
import java.net.URLEncoder
import java.net.http.HttpClient
import java.net.http.HttpRequest
import java.net.http.HttpResponse
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import java.nio.file.StandardOpenOption
import java.security.MessageDigest
import java.time.Duration
import kotlin.io.path.name

data class ModelInstallState(
    val cacheRoot: Path,
    val targetDirectory: Path,
    val assistantDirectory: Path,
    val targetReady: Boolean,
    val assistantReady: Boolean,
    val tokenizerConfigReady: Boolean,
    val isDownloading: Boolean = false,
    val currentFile: String? = null,
    val downloadedBytes: Long = 0,
    val totalBytes: Long = Gem16ModelCatalog.totalBytes,
    val error: String? = null,
) {
    val allReady: Boolean get() = targetReady && assistantReady && tokenizerConfigReady
    val progress: Float get() = if (totalBytes == 0L) 0f else {
        (downloadedBytes.toDouble() / totalBytes.toDouble()).coerceIn(0.0, 1.0).toFloat()
    }
}

class ModelManager {
    private val http = HttpClient.newBuilder()
        .followRedirects(HttpClient.Redirect.NORMAL)
        .connectTimeout(Duration.ofSeconds(30))
        .build()
    private val _state = MutableStateFlow(inspect())
    val state: StateFlow<ModelInstallState> = _state.asStateFlow()

    @Volatile
    private var cancelRequested = false

    fun refresh() {
        if (!_state.value.isDownloading) _state.value = inspect()
    }

    fun cancel() {
        cancelRequested = true
    }

    suspend fun downloadAll(explicitToken: String? = null): ModelInstallState = withContext(Dispatchers.IO) {
        check(!_state.value.isDownloading) { "A model download is already running" }
        cancelRequested = false
        val token = resolveToken(explicitToken)
        var completedBytes = cachedBytes()
        _state.value = inspect().copy(
            isDownloading = true,
            downloadedBytes = completedBytes,
            totalBytes = Gem16ModelCatalog.totalBytes,
            error = null,
        )
        try {
            for (model in listOf(Gem16ModelCatalog.target, Gem16ModelCatalog.assistant)) {
                for (file in model.files) {
                    ensureNotCancelled()
                    val alreadyCached = cachedFileBytes(model, file)
                    ensureCached(model, file, token, completedBytes)
                    if (alreadyCached == 0L) completedBytes += file.size
                    _state.value = _state.value.copy(downloadedBytes = completedBytes)
                }
            }
            composeTargetView()
            val result = inspect()
            check(result.allReady) { "The Hugging Face cache is incomplete after download" }
            _state.value = result
            result
        } catch (error: Exception) {
            val message = if (cancelRequested) "Download paused" else error.message ?: "Model download failed"
            _state.value = inspect().copy(error = message)
            throw error
        }
    }

    private fun inspect(): ModelInstallState {
        val targetDirectory = HuggingFaceCachePaths.targetView()
        val assistantDirectory = HuggingFaceCachePaths.snapshot(
            Gem16ModelCatalog.assistantRepository,
            Gem16ModelCatalog.assistantRevision,
        )
        val tokenizerPath = HuggingFaceCachePaths.snapshot(
            Gem16ModelCatalog.tokenizerRepository,
            Gem16ModelCatalog.tokenizerRevision,
        ).resolve("tokenizer_config.json")
        return ModelInstallState(
            cacheRoot = HuggingFaceCachePaths.hubRoot(),
            targetDirectory = targetDirectory,
            assistantDirectory = assistantDirectory,
            targetReady = Gem16ModelCatalog.target.files.all { fileMatches(targetDirectory.resolve(it.path), it) },
            assistantReady = Gem16ModelCatalog.assistant.files.all {
                fileMatches(assistantDirectory.resolve(it.path), it)
            },
            tokenizerConfigReady = fileMatches(
                tokenizerPath,
                Gem16ModelCatalog.target.files.single { it.path == "tokenizer_config.json" },
            ),
            downloadedBytes = cachedBytes(),
        )
    }

    private fun cachedBytes(): Long = listOf(Gem16ModelCatalog.target, Gem16ModelCatalog.assistant)
        .sumOf { model -> model.files.sumOf { cachedFileBytes(model, it) } }

    private fun cachedFileBytes(model: LockedModel, file: LockedModelFile): Long {
        val source = file.source(model.repository, model.revision)
        val blob = HuggingFaceCachePaths.blob(source.repository, file.blobId)
        return if (fileMatches(blob, file)) file.size else 0L
    }

    private fun ensureCached(
        model: LockedModel,
        file: LockedModelFile,
        token: String?,
        completedBytes: Long,
    ) {
        val source = file.source(model.repository, model.revision)
        val blob = HuggingFaceCachePaths.blob(source.repository, file.blobId)
        Files.createDirectories(blob.parent)
        if (!verified(blob, file)) downloadBlob(source, blob, file, token, completedBytes)
        linkIntoSnapshot(source, blob, file)
    }

    private fun verified(blob: Path, file: LockedModelFile): Boolean {
        if (!fileMatches(blob, file)) return false
        val marker = verificationMarker(blob, file)
        if (Files.isRegularFile(marker) && Files.readString(marker).trim() == file.sha256) return true
        val actual = sha256(blob)
        if (actual != file.sha256) {
            Files.deleteIfExists(blob)
            Files.deleteIfExists(marker)
            return false
        }
        writeVerificationMarker(marker, file.sha256)
        return true
    }

    private fun downloadBlob(
        source: HuggingFaceSource,
        blob: Path,
        file: LockedModelFile,
        token: String?,
        completedBytes: Long,
    ) {
        val partial = blob.resolveSibling("${blob.fileName}.incomplete")
        var offset = if (Files.isRegularFile(partial)) Files.size(partial) else 0L
        if (offset > file.size) {
            Files.delete(partial)
            offset = 0L
        }
        _state.value = _state.value.copy(
            currentFile = "${source.repository}/${source.path}",
            downloadedBytes = completedBytes + offset,
        )
        if (offset == file.size) {
            finishPartialDownload(source, partial, blob, file)
            return
        }
        val requestBuilder = HttpRequest.newBuilder(downloadUri(source))
            .timeout(Duration.ofHours(6))
            .header("User-Agent", "gem16/1")
            .header("Accept-Encoding", "identity")
            .GET()
        if (!token.isNullOrBlank()) requestBuilder.header("Authorization", "Bearer $token")
        if (offset != 0L) requestBuilder.header("Range", "bytes=$offset-")
        val response = http.send(requestBuilder.build(), HttpResponse.BodyHandlers.ofInputStream())
        if (response.statusCode() == 401 || response.statusCode() == 403) {
            response.body().close()
            error(
                "Hugging Face denied access to ${source.repository}. Accept the repository license and " +
                    "sign in with HF_TOKEN or the token field in gem16.",
            )
        }
        check(response.statusCode() == 200 || response.statusCode() == 206) {
            response.body().close()
            "Hugging Face returned HTTP ${response.statusCode()} for ${source.repository}/${source.path}"
        }
        if (response.statusCode() == 206) {
            val contentRange = response.headers().firstValue("Content-Range").orElse("")
            check(contentRange.startsWith("bytes $offset-")) {
                response.body().close()
                "Unexpected Content-Range '$contentRange' while resuming ${source.path} at byte $offset"
            }
        }
        if (offset != 0L && response.statusCode() != 206) {
            Files.deleteIfExists(partial)
            offset = 0L
        }
        val options = if (offset == 0L) {
            arrayOf(StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)
        } else {
            arrayOf(StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.APPEND)
        }
        response.body().use { input ->
            Files.newOutputStream(partial, *options).use { output ->
                val buffer = ByteArray(8 * 1024 * 1024)
                var received = offset
                while (true) {
                    ensureNotCancelled()
                    val count = input.read(buffer)
                    if (count < 0) break
                    output.write(buffer, 0, count)
                    received += count
                    _state.value = _state.value.copy(downloadedBytes = completedBytes + received)
                }
            }
        }
        finishPartialDownload(source, partial, blob, file)
    }

    private fun finishPartialDownload(
        source: HuggingFaceSource,
        partial: Path,
        blob: Path,
        file: LockedModelFile,
    ) {
        check(Files.size(partial) == file.size) {
            "Size mismatch for ${source.path}: expected ${file.size}, got ${Files.size(partial)}"
        }
        val actual = sha256(partial)
        check(actual == file.sha256) {
            Files.deleteIfExists(partial)
            "SHA-256 mismatch for ${source.path}: expected ${file.sha256}, got $actual"
        }
        runCatching {
            Files.move(partial, blob, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE)
        }.getOrElse {
            Files.move(partial, blob, StandardCopyOption.REPLACE_EXISTING)
        }
        writeVerificationMarker(verificationMarker(blob, file), file.sha256)
        linkIntoSnapshot(source, blob, file)
    }

    private fun linkIntoSnapshot(source: HuggingFaceSource, blob: Path, file: LockedModelFile) {
        val target = HuggingFaceCachePaths.snapshot(source.repository, source.revision).resolve(source.path)
        linkFile(blob, target, file.size)
    }

    private fun composeTargetView() {
        val target = HuggingFaceCachePaths.targetView()
        for (file in Gem16ModelCatalog.target.files) {
            val source = file.source(Gem16ModelCatalog.target.repository, Gem16ModelCatalog.target.revision)
            val blob = HuggingFaceCachePaths.blob(source.repository, file.blobId)
            check(verified(blob, file)) { "Missing verified cache blob for ${file.path}" }
            linkFile(blob, target.resolve(file.path), file.size)
        }
        Files.writeString(target.resolve(".gem16-ready"), "${Gem16ModelCatalog.targetRevision}\n")
    }

    private fun linkFile(blob: Path, target: Path, expectedSize: Long) {
        if (Files.isRegularFile(target) &&
            Files.size(target) == expectedSize &&
            runCatching { Files.isSameFile(target, blob) }.getOrDefault(false)
        ) {
            return
        }
        Files.createDirectories(target.parent)
        Files.deleteIfExists(target)
        runCatching { Files.createLink(target, blob) }
            .recoverCatching { Files.createSymbolicLink(target, target.parent.relativize(blob)) }
            .getOrElse {
                error(
                    "Could not link ${target.name} into the Hugging Face snapshot. " +
                        "Keep the cache on a filesystem that supports hardlinks or enable Windows Developer Mode.",
                )
            }
    }

    private fun verificationMarker(blob: Path, file: LockedModelFile): Path = blob.parent.parent
        .resolve(".gem16-verified")
        .resolve("${file.blobId}.sha256")

    private fun writeVerificationMarker(marker: Path, hash: String) {
        Files.createDirectories(marker.parent)
        Files.writeString(marker, "$hash\n")
    }

    private fun fileMatches(path: Path, file: LockedModelFile): Boolean =
        Files.isRegularFile(path) && runCatching { Files.size(path) == file.size }.getOrDefault(false)

    private fun sha256(path: Path): String {
        val digest = MessageDigest.getInstance("SHA-256")
        Files.newInputStream(path).use { input ->
            val buffer = ByteArray(8 * 1024 * 1024)
            while (true) {
                ensureNotCancelled()
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun downloadUri(source: HuggingFaceSource): URI {
        val encodedPath = source.path.split('/').joinToString("/") {
            URLEncoder.encode(it, Charsets.UTF_8).replace("+", "%20")
        }
        return URI.create("https://huggingface.co/${source.repository}/resolve/${source.revision}/$encodedPath")
    }

    private fun resolveToken(explicitToken: String?): String? {
        explicitToken?.trim()?.takeIf(String::isNotEmpty)?.let { return it }
        sequenceOf("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN").forEach { name ->
            System.getenv(name)?.trim()?.takeIf(String::isNotEmpty)?.let { return it }
        }
        val hfHome = System.getenv("HF_HOME")?.takeIf(String::isNotBlank)?.let { Path.of(it) }
            ?: Path.of(System.getProperty("user.home"), ".cache", "huggingface")
        val tokenFile = hfHome.resolve("token")
        return tokenFile.takeIf { Files.isRegularFile(it) }?.let {
            runCatching { Files.readString(it).trim().takeIf(String::isNotEmpty) }.getOrNull()
        }
    }

    private fun ensureNotCancelled() {
        if (cancelRequested) throw InterruptedException("Model download paused")
    }
}
