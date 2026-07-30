package com.gem16.studio.service

import com.gem16.studio.model.MediaAttachment
import com.gem16.studio.model.MediaKind
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.extension
import kotlin.io.path.name

const val MaxSingleMediaBytes: Long = 10L * 1024L * 1024L
const val MaxEncodedMediaBytes: Long = 14L * 1024L * 1024L

private data class MediaType(val kind: MediaKind, val mimeType: String, val format: String)

private val mediaTypes = mapOf(
    "png" to MediaType(MediaKind.Image, "image/png", "png"),
    "jpg" to MediaType(MediaKind.Image, "image/jpeg", "jpeg"),
    "jpeg" to MediaType(MediaKind.Image, "image/jpeg", "jpeg"),
    "bmp" to MediaType(MediaKind.Image, "image/bmp", "bmp"),
    "wav" to MediaType(MediaKind.Audio, "audio/wav", "wav"),
    "flac" to MediaType(MediaKind.Audio, "audio/flac", "flac"),
    "mp3" to MediaType(MediaKind.Audio, "audio/mpeg", "mp3"),
)

val supportedMediaExtensions: Set<String> = mediaTypes.keys

fun loadMediaAttachment(path: Path): Result<MediaAttachment> = runCatching {
    val normalized = path.toAbsolutePath().normalize()
    require(Files.isRegularFile(normalized)) { "Media file does not exist: $normalized" }
    val type = mediaTypes[normalized.extension.lowercase()]
        ?: error("Unsupported media type: ${normalized.name}. Use PNG, JPEG, BMP, WAV, FLAC, or MP3.")
    val size = Files.size(normalized)
    require(size > 0L) { "Media file is empty: ${normalized.name}" }
    require(size <= MaxSingleMediaBytes) {
        "${normalized.name} is ${formatBytes(size)}; the Studio limit is ${formatBytes(MaxSingleMediaBytes)}."
    }
    val bytes = Files.readAllBytes(normalized)
    MediaAttachment(
        fileName = normalized.name,
        kind = type.kind,
        mimeType = type.mimeType,
        format = type.format,
        bytes = bytes,
    )
}

fun encodedMediaBytes(messages: List<com.gem16.studio.model.ChatMessage>): Long =
    messages.sumOf { message -> message.attachments.sumOf(MediaAttachment::encodedSize) }

fun formatBytes(value: Long): String = when {
    value >= 1024L * 1024L -> "%.1f MiB".format(value.toDouble() / (1024.0 * 1024.0))
    value >= 1024L -> "%.1f KiB".format(value.toDouble() / 1024.0)
    else -> "$value B"
}
