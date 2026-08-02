package com.gem16.studio

import com.gem16.studio.service.AudioRecorder
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.io.OutputStream
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertSame

class AudioRecorderTest {
    @Test
    fun retriesOneTransientPipeWireStartupFailure() {
        val failed = FakeProcess(alive = false, exitCode = 1)
        val ready = FakeProcess(alive = true, exitCode = 0)
        val processes = ArrayDeque<Process>().apply {
            addLast(failed)
            addLast(ready)
        }
        val commands = mutableListOf<List<String>>()
        val sleeps = mutableListOf<Long>()
        val recorder = AudioRecorder(
            pipeWireProcessFactory = { command ->
                commands += command
                processes.removeFirst()
            },
            sleep = sleeps::add,
        )

        var retryPreparations = 0
        try {
            val selected = recorder.startPipeWireProcess(
                sampleRate = 48_000,
                channels = 1,
                beforeRetry = { retryPreparations++ },
            )

            assertSame(ready, selected)
            assertEquals(1, retryPreparations)
            assertEquals(2, commands.size)
            assertEquals(commands[0], commands[1])
            assertEquals(
                listOf(
                    "pw-record",
                    "--rate", "48000",
                    "--channels", "1",
                    "--format", "s16",
                    "--raw",
                    "-",
                ),
                commands[0],
            )
            assertEquals(listOf(150L, 500L, 150L), sleeps)
        } finally {
            recorder.close()
        }
        assertFalse(ready.isAlive)
    }

    @Test
    fun reportsFailureAfterTheBoundedPipeWireRetry() {
        var starts = 0
        val recorder = AudioRecorder(
            pipeWireProcessFactory = {
                starts++
                FakeProcess(alive = false, exitCode = 1)
            },
            sleep = {},
        )

        val error = assertFailsWith<IllegalStateException> {
            recorder.startPipeWireProcess(sampleRate = 48_000, channels = 1)
        }

        assertEquals(2, starts)
        assertEquals("PipeWire microphone recording could not be started", error.message)
    }

    @Test
    fun cancellationDuringRetryDelayDoesNotStartAnotherProcess() {
        var starts = 0
        var retryPreparations = 0
        lateinit var recorder: AudioRecorder
        recorder = AudioRecorder(
            pipeWireProcessFactory = {
                starts++
                FakeProcess(alive = false, exitCode = 1)
            },
            sleep = { delay ->
                if (delay == 500L) recorder.stop()
            },
        )

        assertFailsWith<IllegalStateException> {
            recorder.startPipeWireProcess(
                sampleRate = 48_000,
                channels = 1,
                beforeRetry = { retryPreparations++ },
            )
        }

        assertEquals(1, starts)
        assertEquals(0, retryPreparations)
    }

    private class FakeProcess(
        @Volatile private var alive: Boolean,
        private val exitCode: Int,
    ) : Process() {
        override fun getOutputStream(): OutputStream = ByteArrayOutputStream()
        override fun getInputStream(): InputStream = ByteArrayInputStream(ByteArray(0))
        override fun getErrorStream(): InputStream = ByteArrayInputStream(ByteArray(0))

        override fun waitFor(): Int {
            alive = false
            return exitCode
        }

        override fun exitValue(): Int {
            if (alive) throw IllegalThreadStateException("process is still running")
            return exitCode
        }

        override fun destroy() {
            alive = false
        }

        override fun isAlive(): Boolean = alive
    }
}
