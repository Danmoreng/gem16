package com.gem16.studio

import com.gem16.studio.service.awaitPipeWireSourceVolume
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class AudioRecorderTest {
    @Test
    fun readyPipeWireSourceReturnsWithoutPolling() {
        var queries = 0
        val sleeps = mutableListOf<Long>()

        val volume = awaitPipeWireSourceVolume(
            timeoutMillis = 2_000L,
            pollMillis = 100L,
            stopRequested = { false },
            query = { remainingMillis ->
                assertEquals(2_000L, remainingMillis)
                queries++
                0.75f
            },
            nanoTime = { 0L },
            sleep = sleeps::add,
        )

        assertEquals(0.75f, volume)
        assertEquals(1, queries)
        assertEquals(emptyList(), sleeps)
    }

    @Test
    fun waitsOnlyUntilWirePlumberPublishesTheDefaultSource() {
        var nowNanos = 0L
        var queries = 0
        val sleeps = mutableListOf<Long>()
        val queryBudgets = mutableListOf<Long>()

        val volume = awaitPipeWireSourceVolume(
            timeoutMillis = 2_000L,
            pollMillis = 100L,
            stopRequested = { false },
            query = { remainingMillis ->
                queryBudgets += remainingMillis
                queries++
                if (queries == 3) 1.0f else null
            },
            nanoTime = { nowNanos },
            sleep = { millis ->
                sleeps += millis
                nowNanos += millis * 1_000_000L
            },
        )

        assertEquals(1.0f, volume)
        assertEquals(3, queries)
        assertEquals(listOf(100L, 100L), sleeps)
        assertEquals(listOf(2_000L, 1_900L, 1_800L), queryBudgets)
        assertEquals(200_000_000L, nowNanos)
    }

    @Test
    fun stopsPollingAtTheTwoSecondDeadline() {
        var nowNanos = 0L
        var queries = 0

        val volume = awaitPipeWireSourceVolume(
            timeoutMillis = 2_000L,
            pollMillis = 100L,
            stopRequested = { false },
            query = { _ ->
                queries++
                null
            },
            nanoTime = { nowNanos },
            sleep = { millis -> nowNanos += millis * 1_000_000L },
        )

        assertNull(volume)
        assertEquals(20, queries)
        assertEquals(2_000_000_000L, nowNanos)
    }

    @Test
    fun cancellationStopsReadinessPolling() {
        var nowNanos = 0L
        var cancelled = false
        var queries = 0

        val volume = awaitPipeWireSourceVolume(
            timeoutMillis = 2_000L,
            pollMillis = 100L,
            stopRequested = { cancelled },
            query = { _ ->
                queries++
                null
            },
            nanoTime = { nowNanos },
            sleep = { millis ->
                nowNanos += millis * 1_000_000L
                cancelled = true
            },
        )

        assertNull(volume)
        assertEquals(1, queries)
        assertEquals(100_000_000L, nowNanos)
    }
}
