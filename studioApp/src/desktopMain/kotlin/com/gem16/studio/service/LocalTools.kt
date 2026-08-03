package com.gem16.studio.service

import com.gem16.studio.model.ToolCall
import java.time.Clock
import java.time.LocalDate
import java.time.ZonedDateTime
import java.time.format.DateTimeFormatter
import java.util.Locale
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put

const val MaxLocalToolRounds: Int = 8

class LocalToolExecutor(
    private val clock: Clock = Clock.systemDefaultZone(),
) {
    fun execute(call: ToolCall): String = runCatching {
        requireNoArguments(call)
        when (call.name) {
            "get_current_date" -> currentDate()
            "get_current_time" -> currentTime()
            else -> error("Unknown local tool: ${call.name}")
        }
    }.getOrElse { error ->
        jsonObject(
            "ok" to JsonPrimitive(false),
            "error" to JsonPrimitive(error.message ?: "Tool execution failed"),
        ).toString()
    }

    private fun currentDate(): String {
        val now = ZonedDateTime.now(clock)
        return jsonObject(
            "ok" to JsonPrimitive(true),
            "date" to JsonPrimitive(LocalDate.from(now).toString()),
            "day_of_week" to JsonPrimitive(now.dayOfWeek.name.lowercase(Locale.ROOT)),
            "timezone" to JsonPrimitive(now.zone.id),
        ).toString()
    }

    private fun currentTime(): String {
        val now = ZonedDateTime.now(clock)
        return jsonObject(
            "ok" to JsonPrimitive(true),
            "time" to JsonPrimitive(now.format(DateTimeFormatter.ofPattern("HH:mm:ss"))),
            "utc_offset" to JsonPrimitive(now.offset.id),
            "timezone" to JsonPrimitive(now.zone.id),
            "iso_datetime" to JsonPrimitive(now.format(DateTimeFormatter.ISO_OFFSET_DATE_TIME)),
        ).toString()
    }

    private fun requireNoArguments(call: ToolCall) {
        val value = call.argumentsJson.ifBlank { "{}" }
        val arguments = runCatching { Json.parseToJsonElement(value).jsonObject }
            .getOrElse { throw IllegalArgumentException("Tool arguments must be a JSON object.") }
        require(arguments.isEmpty()) { "${call.name} does not accept arguments." }
    }
}

private fun jsonObject(vararg fields: Pair<String, JsonPrimitive>): JsonObject = buildJsonObject {
    fields.forEach { (key, value) -> put(key, value) }
}
