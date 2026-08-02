package com.gem16.studio.service

// PipeWire fallback, mixer ranking, and signal-validation policy adapted from
// qwen-tts-studio VoicesViewModel.kt at ef2344a702ea056e549dac2fbb6c961b57b5feb2 (MIT).

import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.TimeUnit
import javax.sound.sampled.AudioFormat
import javax.sound.sampled.AudioSystem
import javax.sound.sampled.DataLine
import javax.sound.sampled.TargetDataLine
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

const val MaxRecordingSeconds: Int = 30

private const val MinimumRecordingMillis = 250L
private const val SilencePeakThreshold = 128
private const val SilenceRmsThreshold = 8.0
private const val ClippingSampleThreshold = 0.01
private const val PipeWireRecordingVolume = 0.05f
private const val PipeWireStartupProbeMillis = 150L
private const val PipeWireReadinessTimeoutMillis = 2_000L
private const val PipeWireReadinessPollMillis = 100L
private const val PipeWireControlAttemptMillis = 250L
private const val NormalizedPeak = 27_852

internal data class RecordedAudio(
    val fileName: String,
    val wavBytes: ByteArray,
    val durationMillis: Long,
    val sampleRate: Int,
    val channels: Int,
)

private data class PcmStats(
    val peak: Int,
    val squareSum: Double,
    val sampleCount: Long,
    val clippedSamples: Long,
) {
    val rms: Double get() = if (sampleCount == 0L) 0.0 else sqrt(squareSum / sampleCount)
    val clippingRatio: Double get() =
        if (sampleCount == 0L) 0.0 else clippedSamples.toDouble() / sampleCount.toDouble()
}

class AudioRecorder : AutoCloseable {
    @Volatile
    private var stopRequested = false
    @Volatile
    private var activeLine: TargetDataLine? = null
    @Volatile
    private var activeProcess: Process? = null

    internal fun record(onProgress: (durationMillis: Long, level: Float) -> Unit): RecordedAudio {
        check(activeLine == null && activeProcess == null) { "An audio recording is already active" }
        stopRequested = false
        return if (
            isLinux() && executableOnPath("pw-record") && executableOnPath("wpctl")
        ) {
            recordPipeWire(onProgress)
        } else {
            recordJavaSound(onProgress)
        }
    }

    fun stop() {
        stopRequested = true
        activeLine?.let { line ->
            runCatching { line.stop() }
            runCatching { line.close() }
        }
        activeProcess?.let { process -> runCatching { process.destroy() } }
    }

    override fun close() = stop()

    private fun recordPipeWire(onProgress: (Long, Float) -> Unit): RecordedAudio {
        val sampleRate = 48_000
        val channels = 1
        val previousVolume = lowerDefaultPipeWireSourceVolume()
        val process = try {
            startPipeWireProcess(sampleRate, channels)
        } catch (error: Exception) {
            restoreDefaultPipeWireSourceVolume(previousVolume)
            throw error
        }
        try {
            val pcm = capturePcm(process.inputStream, sampleRate, channels, onProgress)
            return finishRecording(pcm, sampleRate, channels)
        } finally {
            stopProcess(process)
            if (activeProcess === process) activeProcess = null
            restoreDefaultPipeWireSourceVolume(previousVolume)
        }
    }

    private fun startPipeWireProcess(sampleRate: Int, channels: Int): Process {
        val process = launchPipeWireProcess(
            listOf(
                "pw-record",
                "--rate", sampleRate.toString(),
                "--channels", channels.toString(),
                "--format", "s16",
                "--raw",
                "-",
            ),
        )
        activeProcess = process
        try {
            Thread.sleep(PipeWireStartupProbeMillis)
            check(process.isAlive) { "PipeWire microphone recording could not be started" }
            return process
        } catch (error: Throwable) {
            stopProcess(process)
            if (activeProcess === process) activeProcess = null
            throw error
        }
    }

    private fun recordJavaSound(onProgress: (Long, Float) -> Unit): RecordedAudio {
        val (line, format) = openCaptureLine()
        activeLine = line
        try {
            line.start()
            val pcm = capturePcm(
                input = object : InputStream() {
                    override fun read(): Int = error("single-byte microphone reads are unsupported")
                    override fun read(bytes: ByteArray, offset: Int, length: Int): Int =
                        try {
                            line.read(bytes, offset, length)
                        } catch (error: Exception) {
                            if (stopRequested) -1 else throw error
                        }
                },
                sampleRate = format.sampleRate.toInt(),
                channels = format.channels,
                onProgress = onProgress,
            )
            return finishRecording(pcm, format.sampleRate.toInt(), format.channels)
        } finally {
            runCatching { line.stop() }
            runCatching { line.flush() }
            runCatching { line.close() }
            activeLine = null
        }
    }

    private fun capturePcm(
        input: InputStream,
        sampleRate: Int,
        channels: Int,
        onProgress: (Long, Float) -> Unit,
    ): ByteArray = capturePcmStream(
        input = input,
        sampleRate = sampleRate,
        channels = channels,
        maximumSeconds = MaxRecordingSeconds,
        stopRequested = { stopRequested },
        onProgress = onProgress,
    )

    private fun finishRecording(pcm: ByteArray, sampleRate: Int, channels: Int): RecordedAudio {
        val frameSize = channels * 2
        val alignedSize = pcm.size - (pcm.size % frameSize)
        require(alignedSize > 0) { "The microphone did not return any audio" }
        val alignedPcm = if (alignedSize == pcm.size) pcm else pcm.copyOf(alignedSize)
        val duration = durationMillis(alignedPcm.size, sampleRate, frameSize)
        require(duration >= MinimumRecordingMillis) { "Recording was too short" }
        val normalized = prepareRecordedPcm16(alignedPcm)
        return RecordedAudio(
            fileName = "recording-${LocalDateTime.now().format(FileNameTime)}.wav",
            wavBytes = encodePcm16Wav(normalized, sampleRate, channels),
            durationMillis = duration,
            sampleRate = sampleRate,
            channels = channels,
        )
    }

    private fun openCaptureLine(): Pair<TargetDataLine, AudioFormat> {
        val candidates = listOf(
            captureFormat(44_100f, 1),
            captureFormat(48_000f, 1),
            captureFormat(24_000f, 1),
            captureFormat(16_000f, 1),
            captureFormat(48_000f, 2),
            captureFormat(44_100f, 2),
        )
        val mixers = AudioSystem.getMixerInfo().sortedByDescending(::mixerRank)
        var lastError: Exception? = null
        for (format in candidates) {
            val info = DataLine.Info(TargetDataLine::class.java, format)
            for (mixerInfo in mixers) {
                try {
                    val mixer = AudioSystem.getMixer(mixerInfo)
                    if (!mixer.isLineSupported(info)) continue
                    val line = mixer.getLine(info) as TargetDataLine
                    line.open(format, recordingBufferBytes(format))
                    return line to format
                } catch (error: Exception) {
                    lastError = error
                }
            }
            try {
                if (!AudioSystem.isLineSupported(info)) continue
                val line = AudioSystem.getLine(info) as TargetDataLine
                line.open(format, recordingBufferBytes(format))
                return line to format
            } catch (error: Exception) {
                lastError = error
            }
        }
        throw IllegalStateException(
            "No compatible microphone is available. Allow microphone access and select a system input device.",
            lastError,
        )
    }

    private fun lowerDefaultPipeWireSourceVolume(): Float? {
        if (!executableOnPath("wpctl")) return null
        val previous = awaitPipeWireSourceVolume(
            timeoutMillis = PipeWireReadinessTimeoutMillis,
            pollMillis = PipeWireReadinessPollMillis,
            stopRequested = { stopRequested },
            query = { remainingMillis ->
                queryDefaultPipeWireSourceVolume(
                    min(PipeWireControlAttemptMillis, remainingMillis),
                )
            },
        ) ?: throw IllegalStateException(
            if (stopRequested) {
                "PipeWire microphone startup was cancelled"
            } else {
                "PipeWire microphone source did not become ready within 2 seconds"
            },
        )
        if (previous > PipeWireRecordingVolume) setPipeWireSourceVolume(PipeWireRecordingVolume)
        return previous
    }

    private fun queryDefaultPipeWireSourceVolume(waitMillis: Long): Float? = runCatching {
        val process = ProcessBuilder("wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@")
            .redirectError(ProcessBuilder.Redirect.DISCARD)
            .start()
        if (!process.waitFor(waitMillis, TimeUnit.MILLISECONDS)) {
            process.destroy()
            if (process.isAlive) process.destroyForcibly()
            return@runCatching null
        }
        if (process.exitValue() != 0) return@runCatching null
        val output = process.inputStream.bufferedReader().use { it.readText() }
        Regex("""Volume:\s*([0-9]+(?:\.[0-9]+)?)""")
            .find(output)?.groupValues?.getOrNull(1)?.toFloatOrNull()
    }.getOrNull()

    private fun restoreDefaultPipeWireSourceVolume(previous: Float?) {
        if (previous != null) setPipeWireSourceVolume(previous)
    }

    private fun setPipeWireSourceVolume(volume: Float) {
        runCatching {
            ProcessBuilder(
                "wpctl", "set-volume", "@DEFAULT_AUDIO_SOURCE@", volume.coerceIn(0f, 1f).toString(),
            )
                .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                .redirectError(ProcessBuilder.Redirect.DISCARD)
                .start()
                .waitFor(1, TimeUnit.SECONDS)
        }
    }
}

internal fun awaitPipeWireSourceVolume(
    timeoutMillis: Long,
    pollMillis: Long,
    stopRequested: () -> Boolean,
    query: (remainingMillis: Long) -> Float?,
    nanoTime: () -> Long = System::nanoTime,
    sleep: (Long) -> Unit = Thread::sleep,
): Float? {
    require(timeoutMillis > 0L && pollMillis > 0L)
    val timeoutNanos = timeoutMillis * 1_000_000L
    val deadline = nanoTime() + timeoutNanos
    while (!stopRequested()) {
        val queryRemainingNanos = deadline - nanoTime()
        if (queryRemainingNanos <= 0L) break
        val queryRemainingMillis =
            (queryRemainingNanos + 999_999L) / 1_000_000L
        query(queryRemainingMillis)?.let { return it }
        val sleepRemainingNanos = deadline - nanoTime()
        if (sleepRemainingNanos <= 0L) break
        val sleepRemainingMillis =
            (sleepRemainingNanos + 999_999L) / 1_000_000L
        sleep(min(pollMillis, sleepRemainingMillis))
    }
    return null
}

internal fun capturePcmStream(
    input: InputStream,
    sampleRate: Int,
    channels: Int,
    maximumSeconds: Int,
    stopRequested: () -> Boolean,
    onProgress: (Long, Float) -> Unit,
): ByteArray {
    require(sampleRate > 0 && channels in 1..2 && maximumSeconds > 0)
    val frameSize = channels * 2
    val maximumBytes = sampleRate.toLong() * frameSize.toLong() * maximumSeconds.toLong()
    val output = ByteArrayOutputStream(min(maximumBytes, 1024L * 1024L).toInt())
    val buffer = ByteArray(16 * 1024)
    var lastProgressNanos = 0L
    while (!stopRequested() && output.size().toLong() < maximumBytes) {
        val remaining = maximumBytes - output.size().toLong()
        val requested = min(buffer.size.toLong(), remaining).toInt()
        val read = try {
            input.read(buffer, 0, requested)
        } catch (error: Exception) {
            if (stopRequested()) break else throw error
        }
        if (read < 0) break
        if (read == 0) continue
        output.write(buffer, 0, read)
        val now = System.nanoTime()
        if (now - lastProgressNanos >= 50_000_000L || output.size().toLong() >= maximumBytes) {
            onProgress(
                durationMillis(output.size(), sampleRate, frameSize),
                pcm16Stats(buffer, read).peak / 32768f,
            )
            lastProgressNanos = now
        }
    }
    return output.toByteArray()
}

private val FileNameTime: DateTimeFormatter = DateTimeFormatter.ofPattern("yyyyMMdd-HHmmss")

private fun captureFormat(sampleRate: Float, channels: Int): AudioFormat = AudioFormat(
    AudioFormat.Encoding.PCM_SIGNED,
    sampleRate,
    16,
    channels,
    channels * 2,
    sampleRate,
    false,
)

private fun durationMillis(byteCount: Int, sampleRate: Int, frameSize: Int): Long =
    byteCount.toLong() * 1000L / (sampleRate.toLong() * frameSize.toLong())

private fun pcm16Stats(bytes: ByteArray, length: Int): PcmStats {
    var peak = 0
    var squareSum = 0.0
    var sampleCount = 0L
    var clippedSamples = 0L
    var index = 0
    while (index + 1 < length) {
        val sample = ((bytes[index + 1].toInt() shl 8) or (bytes[index].toInt() and 0xFF)).toShort().toInt()
        val magnitude = abs(sample)
        peak = maxOf(peak, magnitude)
        squareSum += sample.toDouble() * sample.toDouble()
        sampleCount++
        if (magnitude >= 32_760) clippedSamples++
        index += 2
    }
    return PcmStats(peak, squareSum, sampleCount, clippedSamples)
}

internal fun prepareRecordedPcm16(pcm: ByteArray): ByteArray {
    val stats = pcm16Stats(pcm, pcm.size)
    require(!(stats.peak < SilencePeakThreshold && stats.rms < SilenceRmsThreshold)) {
        "Recording was silent. Check the system microphone input and capture level."
    }
    require(stats.clippingRatio <= ClippingSampleThreshold) {
        "Recording was clipping. Lower the system microphone input level and try again."
    }
    return normalizePcm16(pcm, stats.peak)
}

private fun normalizePcm16(pcm: ByteArray, peak: Int): ByteArray {
    if (peak <= 0) return pcm
    val gain = (NormalizedPeak.toDouble() / peak.toDouble()).coerceIn(0.25, 16.0)
    if (gain in 0.98..1.02) return pcm
    val output = pcm.copyOf()
    var index = 0
    while (index + 1 < output.size) {
        val sample = ((output[index + 1].toInt() shl 8) or (output[index].toInt() and 0xFF)).toShort().toInt()
        val scaled = (sample * gain).roundToInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
        output[index] = scaled.toByte()
        output[index + 1] = (scaled ushr 8).toByte()
        index += 2
    }
    return output
}

private fun mixerRank(info: javax.sound.sampled.Mixer.Info): Int {
    val text = "${info.name} ${info.description}".lowercase()
    var score = 0
    if (text.contains("generic") || text.contains("analog")) score += 100
    if (text.contains("mic") || text.contains("capture") || text.contains("input")) score += 50
    if (text.contains("default") || text.contains("pulse") || text.contains("pipewire")) score += 40
    if (text.contains("monitor") || text.contains("loopback")) score -= 100
    if (text.contains("hdmi") || text.contains("nvidia")) score -= 50
    if (text.contains("port ")) score -= 25
    return score
}

private fun recordingBufferBytes(format: AudioFormat): Int {
    val frames = (format.frameRate * 100 / 1000).toInt().coerceAtLeast(1024)
    return frames * format.frameSize.coerceAtLeast(1)
}

private fun stopProcess(process: Process) {
    if (!process.isAlive) return
    process.destroy()
    if (!runCatching { process.waitFor(1, TimeUnit.SECONDS) }.getOrDefault(false)) {
        process.destroyForcibly()
        runCatching { process.waitFor(1, TimeUnit.SECONDS) }
    }
}

private fun launchPipeWireProcess(command: List<String>): Process =
    ProcessBuilder(command)
        .redirectError(ProcessBuilder.Redirect.DISCARD)
        .start()

private fun isLinux(): Boolean = System.getProperty("os.name").contains("linux", ignoreCase = true)

private fun executableOnPath(name: String): Boolean =
    System.getenv("PATH")?.split(File.pathSeparator)?.any { File(it, name).canExecute() } == true

internal fun encodePcm16Wav(pcm: ByteArray, sampleRate: Int, channels: Int): ByteArray {
    require(sampleRate > 0) { "sample rate must be positive" }
    require(channels == 1 || channels == 2) { "only mono and stereo PCM are supported" }
    require(pcm.size % (channels * 2) == 0) { "PCM byte count must contain complete frames" }
    val output = ByteArray(44 + pcm.size)
    fun ascii(offset: Int, value: String) {
        value.toByteArray(Charsets.US_ASCII).copyInto(output, offset)
    }
    fun little16(offset: Int, value: Int) {
        output[offset] = value.toByte()
        output[offset + 1] = (value ushr 8).toByte()
    }
    fun little32(offset: Int, value: Int) {
        output[offset] = value.toByte()
        output[offset + 1] = (value ushr 8).toByte()
        output[offset + 2] = (value ushr 16).toByte()
        output[offset + 3] = (value ushr 24).toByte()
    }
    ascii(0, "RIFF")
    little32(4, 36 + pcm.size)
    ascii(8, "WAVE")
    ascii(12, "fmt ")
    little32(16, 16)
    little16(20, 1)
    little16(22, channels)
    little32(24, sampleRate)
    little32(28, sampleRate * channels * 2)
    little16(32, channels * 2)
    little16(34, 16)
    ascii(36, "data")
    little32(40, pcm.size)
    pcm.copyInto(output, 44)
    return output
}
