package com.gem16.studio.service

import com.gem16.studio.model.MediaAttachment
import com.gem16.studio.model.MediaKind
import java.awt.Image
import java.awt.Toolkit
import java.awt.datatransfer.DataFlavor
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.nio.file.Files
import java.nio.file.Path
import javax.imageio.ImageIO
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
        "${normalized.name} is ${formatBytes(size)}; the gem16 limit is ${formatBytes(MaxSingleMediaBytes)}."
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

fun clipboardContainsImage(): Boolean = runCatching {
    Toolkit.getDefaultToolkit().systemClipboard.isDataFlavorAvailable(DataFlavor.imageFlavor)
}.getOrDefault(false)

fun loadClipboardImageAttachment(): Result<MediaAttachment> = runCatching {
    val clipboard = Toolkit.getDefaultToolkit().systemClipboard
    require(clipboard.isDataFlavorAvailable(DataFlavor.imageFlavor)) {
        "The clipboard does not contain an image."
    }
    val image = clipboard.getData(DataFlavor.imageFlavor) as? Image
        ?: error("The clipboard image could not be decoded.")
    encodeClipboardImage(image)
}

internal fun encodeClipboardImage(image: Image): MediaAttachment {
    val width = image.getWidth(null)
    val height = image.getHeight(null)
    require(width > 0 && height > 0) { "The clipboard image has invalid dimensions." }
    val buffered = if (image is BufferedImage && image.type != BufferedImage.TYPE_CUSTOM) {
        image
    } else {
        BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB).also { destination ->
            val graphics = destination.createGraphics()
            try {
                graphics.drawImage(image, 0, 0, null)
            } finally {
                graphics.dispose()
            }
        }
    }
    val bytes = ByteArrayOutputStream().use { output ->
        check(ImageIO.write(buffered, "png", output)) { "PNG encoding is unavailable." }
        output.toByteArray()
    }
    require(bytes.isNotEmpty()) { "The clipboard image is empty." }
    require(bytes.size.toLong() <= MaxSingleMediaBytes) {
        "The clipboard image is ${formatBytes(bytes.size.toLong())}; " +
            "the gem16 limit is ${formatBytes(MaxSingleMediaBytes)}."
    }
    return MediaAttachment(
        fileName = "clipboard-image.png",
        kind = MediaKind.Image,
        mimeType = "image/png",
        format = "png",
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
