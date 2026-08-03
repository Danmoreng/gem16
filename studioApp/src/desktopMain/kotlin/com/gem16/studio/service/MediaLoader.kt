package com.gem16.studio.service

import com.gem16.studio.model.MediaAttachment
import com.gem16.studio.model.MediaKind
import java.awt.Image
import java.awt.Toolkit
import java.awt.datatransfer.DataFlavor
import java.awt.image.BufferedImage
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction
import java.nio.file.Files
import java.nio.file.Path
import javax.imageio.ImageIO
import kotlin.io.path.extension
import kotlin.io.path.name
import org.apache.pdfbox.Loader
import org.apache.pdfbox.text.PDFTextStripper

const val MaxSingleMediaBytes: Long = 10L * 1024L * 1024L
const val MaxEncodedMediaBytes: Long = 14L * 1024L * 1024L
const val MaxDocumentCharacters: Int = 500_000

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

private val textDocumentExtensions = setOf(
    "txt", "md", "markdown", "csv", "tsv", "json", "jsonl", "xml", "yaml", "yml", "log",
    "kt", "kts", "java", "py", "c", "cc", "cpp", "h", "hpp", "js", "jsx", "ts", "tsx",
    "html", "htm", "css", "sql", "toml", "ini", "properties",
)

val supportedMediaExtensions: Set<String> = mediaTypes.keys + textDocumentExtensions + "pdf"

fun loadMediaAttachment(path: Path): Result<MediaAttachment> = runCatching {
    val normalized = path.toAbsolutePath().normalize()
    require(Files.isRegularFile(normalized)) { "Attachment does not exist: $normalized" }
    val extension = normalized.extension.lowercase()
    val size = Files.size(normalized)
    require(size > 0L) { "Attachment is empty: ${normalized.name}" }
    require(size <= MaxSingleMediaBytes) {
        "${normalized.name} is ${formatBytes(size)}; the gem16 limit is ${formatBytes(MaxSingleMediaBytes)}."
    }
    if (extension == "pdf") return@runCatching loadPdfDocument(normalized, size)
    if (extension in textDocumentExtensions) return@runCatching loadTextDocument(normalized, size)
    val type = mediaTypes[extension]
        ?: error(
            "Unsupported attachment type: ${normalized.name}. Use text, PDF, PNG, JPEG, BMP, WAV, FLAC, or MP3.",
        )
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

private fun loadTextDocument(path: Path, size: Long): MediaAttachment {
    val bytes = Files.readAllBytes(path)
    val text = decodeText(bytes, path.name)
    require(text.isNotBlank()) { "${path.name} contains no readable text." }
    require(text.length <= MaxDocumentCharacters) {
        "${path.name} contains ${text.length} characters; the gem16 document limit is " +
            "$MaxDocumentCharacters characters. Split the file into smaller parts."
    }
    return MediaAttachment(
        fileName = path.name,
        kind = MediaKind.Document,
        mimeType = "text/plain",
        format = path.extension.lowercase(),
        bytes = ByteArray(0),
        documentText = text,
        sourceByteSize = size,
    )
}

private fun loadPdfDocument(path: Path, size: Long): MediaAttachment {
    Loader.loadPDF(path.toFile()).use { document ->
        require(!document.isEncrypted) { "Encrypted PDF files are not supported: ${path.name}" }
        val text = normalizeDocumentText(PDFTextStripper().getText(document))
        require(text.isNotBlank()) {
            "${path.name} contains no extractable text. Scanned PDFs need OCR, which is not enabled yet."
        }
        require(text.length <= MaxDocumentCharacters) {
            "${path.name} contains ${text.length} extracted characters; the gem16 document limit is " +
                "$MaxDocumentCharacters characters. Split the PDF into smaller parts."
        }
        return MediaAttachment(
            fileName = path.name,
            kind = MediaKind.Document,
            mimeType = "application/pdf",
            format = "pdf",
            bytes = ByteArray(0),
            documentText = text,
            pageCount = document.numberOfPages,
            sourceByteSize = size,
        )
    }
}

private fun decodeText(bytes: ByteArray, fileName: String): String {
    val (charset, offset) = when {
        bytes.size >= 3 && bytes[0] == 0xEF.toByte() && bytes[1] == 0xBB.toByte() &&
            bytes[2] == 0xBF.toByte() -> Charsets.UTF_8 to 3
        bytes.size >= 2 && bytes[0] == 0xFF.toByte() && bytes[1] == 0xFE.toByte() -> Charsets.UTF_16LE to 2
        bytes.size >= 2 && bytes[0] == 0xFE.toByte() && bytes[1] == 0xFF.toByte() -> Charsets.UTF_16BE to 2
        else -> Charsets.UTF_8 to 0
    }
    val decoded = runCatching {
        charset.newDecoder()
            .onMalformedInput(CodingErrorAction.REPORT)
            .onUnmappableCharacter(CodingErrorAction.REPORT)
            .decode(ByteBuffer.wrap(bytes, offset, bytes.size - offset))
            .toString()
    }.getOrElse {
        throw IllegalArgumentException("$fileName is not valid UTF-8/UTF-16 text.", it)
    }
    return normalizeDocumentText(decoded)
}

private fun normalizeDocumentText(text: String): String =
    text.replace("\u0000", "").replace("\r\n", "\n").replace('\r', '\n').trim()

fun encodedMediaBytes(messages: List<com.gem16.studio.model.ChatMessage>): Long =
    messages.sumOf { message -> message.attachments.sumOf(MediaAttachment::encodedSize) }

fun formatBytes(value: Long): String = when {
    value >= 1024L * 1024L -> "%.1f MiB".format(value.toDouble() / (1024.0 * 1024.0))
    value >= 1024L -> "%.1f KiB".format(value.toDouble() / 1024.0)
    else -> "$value B"
}
