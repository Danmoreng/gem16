package com.gem16.studio

import com.gem16.studio.model.MediaKind
import com.gem16.studio.service.capturePcmStream
import com.gem16.studio.service.encodePcm16Wav
import com.gem16.studio.service.loadMediaAttachment
import com.gem16.studio.service.prepareRecordedPcm16
import java.io.IOException
import java.io.InputStream
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class MediaLoaderTest {
    @Test
    fun expectedStreamClosureRetainsCapturedPcm() {
        var stopped = false
        var reads = 0
        val input = object : InputStream() {
            override fun read(): Int = error("not used")
            override fun read(bytes: ByteArray, offset: Int, length: Int): Int {
                if (reads++ > 0) {
                    stopped = true
                    throw IOException("Stream closed")
                }
                val written = minOf(length, 3_200)
                for (index in 0 until written step 2) {
                    val sample = if ((index / 2) % 2 == 0) 2_000 else -2_000
                    bytes[offset + index] = sample.toByte()
                    bytes[offset + index + 1] = (sample ushr 8).toByte()
                }
                return written
            }
        }
        val captured = capturePcmStream(
            input = input,
            sampleRate = 16_000,
            channels = 1,
            maximumSeconds = 1,
            stopRequested = { stopped },
            onProgress = { _, _ -> },
        )
        assertEquals(3_200, captured.size)
    }

    @Test
    fun unexpectedStreamClosureRemainsAnError() {
        val input = object : InputStream() {
            override fun read(): Int = throw IOException("Stream closed")
            override fun read(bytes: ByteArray, offset: Int, length: Int): Int =
                throw IOException("Stream closed")
        }
        assertFailsWith<IOException> {
            capturePcmStream(input, 16_000, 1, 1, { false }) { _, _ -> }
        }
    }

    @Test
    fun encodesBoundedPcmAsStandardWav() {
        val pcm = byteArrayOf(0, 0, 0xFF.toByte(), 0x7F)
        val wav = encodePcm16Wav(pcm, sampleRate = 16_000, channels = 1)
        assertEquals("RIFF", wav.copyOfRange(0, 4).toString(Charsets.US_ASCII))
        assertEquals("WAVE", wav.copyOfRange(8, 12).toString(Charsets.US_ASCII))
        assertEquals(16_000, little32(wav, 24))
        assertEquals(32_000, little32(wav, 28))
        assertEquals(4, little32(wav, 40))
        assertContentEquals(pcm, wav.copyOfRange(44, wav.size))
        assertFailsWith<IllegalArgumentException> { encodePcm16Wav(byteArrayOf(0), 16_000, 1) }
    }

    @Test
    fun microphonePreparationRejectsSilenceAndClippingAndNormalizesSpeech() {
        assertFailsWith<IllegalArgumentException> {
            prepareRecordedPcm16(ByteArray(2_000))
        }
        val clipped = ByteArray(2_000).also { bytes ->
            for (index in bytes.indices step 2) {
                bytes[index] = 0xFF.toByte()
                bytes[index + 1] = 0x7F
            }
        }
        assertFailsWith<IllegalArgumentException> { prepareRecordedPcm16(clipped) }

        val quietSpeech = ByteArray(2_000).also { bytes ->
            for (index in bytes.indices step 2) {
                val sample = if ((index / 2) % 2 == 0) 2_000 else -2_000
                bytes[index] = sample.toByte()
                bytes[index + 1] = (sample ushr 8).toByte()
            }
        }
        val normalized = prepareRecordedPcm16(quietSpeech)
        assertTrue(normalizedPeak(normalized) in 27_840..27_860)
    }

    @Test
    fun recognizesSupportedMediaAndPreservesBytes() {
        val directory = Files.createTempDirectory("gem16-media-test")
        try {
            val path = directory.resolve("sample.JPEG")
            val source = byteArrayOf(1, 3, 3, 7)
            Files.write(path, source)
            val attachment = loadMediaAttachment(path).getOrThrow()
            assertEquals(MediaKind.Image, attachment.kind)
            assertEquals("image/jpeg", attachment.mimeType)
            assertEquals("jpeg", attachment.format)
            assertEquals(8L, attachment.encodedSize)
            assertContentEquals(source, attachment.bytes)
        } finally {
            directory.toFile().deleteRecursively()
        }
    }

    @Test
    fun rejectsUnsupportedMedia() {
        val file = Files.createTempFile("gem16-media-test", ".txt")
        try {
            Files.writeString(file, "not media")
            val result = loadMediaAttachment(file)
            assertTrue(result.isFailure)
            assertTrue(result.exceptionOrNull()?.message?.contains("Unsupported media type") == true)
        } finally {
            Files.deleteIfExists(file)
        }
    }
}

private fun normalizedPeak(bytes: ByteArray): Int {
    var peak = 0
    for (index in bytes.indices step 2) {
        val sample = ((bytes[index + 1].toInt() shl 8) or (bytes[index].toInt() and 0xFF)).toShort().toInt()
        peak = maxOf(peak, kotlin.math.abs(sample))
    }
    return peak
}

private fun little32(bytes: ByteArray, offset: Int): Int =
    (bytes[offset].toInt() and 0xFF) or
        ((bytes[offset + 1].toInt() and 0xFF) shl 8) or
        ((bytes[offset + 2].toInt() and 0xFF) shl 16) or
        ((bytes[offset + 3].toInt() and 0xFF) shl 24)
