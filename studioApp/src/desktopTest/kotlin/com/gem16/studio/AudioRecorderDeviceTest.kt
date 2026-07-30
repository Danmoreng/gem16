package com.gem16.studio

import com.gem16.studio.service.AudioRecorder
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class AudioRecorderDeviceTest {
    @Test
    fun recordsFromRealDeviceWhenExplicitlyEnabled() {
        if (System.getenv("GEM16_TEST_MIC") != "1") return
        val recorder = AudioRecorder()
        val executor = Executors.newSingleThreadExecutor()
        try {
            val result = executor.submit<com.gem16.studio.service.RecordedAudio> {
                recorder.record { _, _ -> }
            }
            Thread.sleep(1_500)
            recorder.stop()
            val recording = result.get(10, TimeUnit.SECONDS)
            assertTrue(recording.durationMillis >= 1_000)
            assertTrue(recording.wavBytes.size > 44)
            assertEquals("RIFF", recording.wavBytes.copyOfRange(0, 4).toString(Charsets.US_ASCII))
        } finally {
            recorder.close()
            executor.shutdownNow()
        }
    }
}
