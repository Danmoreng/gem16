package com.gem16.studio

import com.gem16.studio.model.ToolCall
import com.gem16.studio.service.LocalToolExecutor
import java.time.Clock
import java.time.Instant
import java.time.ZoneId
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.test.Test
import kotlin.test.assertEquals

class LocalToolExecutorTest {
    private val executor = LocalToolExecutor(
        Clock.fixed(Instant.parse("2026-08-03T10:15:30Z"), ZoneId.of("Europe/Berlin")),
    )

    @Test
    fun returnsDeterministicLocalDateAndTime() {
        val date = result("get_current_date")
        assertEquals("2026-08-03", date["date"]?.jsonPrimitive?.content)
        assertEquals("Europe/Berlin", date["timezone"]?.jsonPrimitive?.content)

        val time = result("get_current_time")
        assertEquals("12:15:30", time["time"]?.jsonPrimitive?.content)
        assertEquals("+02:00", time["utc_offset"]?.jsonPrimitive?.content)
    }

    @Test
    fun rejectsUnknownToolsAndUnexpectedArgumentsAsStructuredErrors() {
        assertEquals("false", result("run_command")["ok"]?.jsonPrimitive?.content)
        val value = executor.execute(ToolCall("call_1", "get_current_time", "{\"extra\":true}"))
        assertEquals(
            "false",
            Json.parseToJsonElement(value).jsonObject["ok"]?.jsonPrimitive?.content,
        )
    }

    private fun result(name: String) = Json.parseToJsonElement(
        executor.execute(ToolCall("call_1", name, "{}")),
    ).jsonObject
}
