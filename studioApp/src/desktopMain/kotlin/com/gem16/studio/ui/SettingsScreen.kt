package com.gem16.studio.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Checkbox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.gem16.studio.state.StudioState
import org.jetbrains.jewel.ui.component.Text

@Composable
fun SettingsScreen(state: StudioState) {
    Row(
        Modifier.fillMaxSize().padding(StudioScreenPadding),
        horizontalArrangement = Arrangement.spacedBy(StudioGap),
    ) {
        StudioSurface(Modifier.weight(1f)) {
            Column(
                Modifier.fillMaxWidth().padding(StudioPanelPadding),
                verticalArrangement = Arrangement.spacedBy(StudioGap),
            ) {
                Text("Generation", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(
                    "Defaults used for new chat responses.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                StudioTextField(
                    value = state.settings.generation.systemPrompt,
                    onValueChange = { prompt ->
                        state.updateGeneration { it.copy(systemPrompt = prompt) }
                    },
                    label = "System prompt",
                    supportingText = "Sent before the conversation. Changes start a fresh resident GPU session.",
                    singleLine = false,
                    minLines = 3,
                    maxLines = 7,
                    modifier = Modifier.fillMaxWidth(),
                )
                var outputText by remember(state.settings.generation.maxOutputTokens) {
                    mutableStateOf(state.settings.generation.maxOutputTokens.toString())
                }
                StudioTextField(
                    value = outputText,
                    onValueChange = { candidate ->
                        if (candidate.all(Char::isDigit)) {
                            outputText = candidate
                            candidate.toLongOrNull()?.takeIf { it in 1L..262144L }?.let { value ->
                                state.updateGeneration { it.copy(maxOutputTokens = value) }
                            }
                        }
                    },
                    label = "Maximum output tokens",
                    supportingText = "Includes private reasoning and the visible answer; it cannot exceed remaining context.",
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(
                        checked = state.settings.generation.showReasoning,
                        onCheckedChange = { checked ->
                            state.updateGeneration { it.copy(showReasoning = checked) }
                        },
                    )
                    Text("Show streamed reasoning in chat")
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(
                        checked = state.settings.generation.localDateTimeTools,
                        onCheckedChange = { checked ->
                            state.updateGeneration { it.copy(localDateTimeTools = checked) }
                        },
                    )
                    Column {
                        Text("Allow local date and time tools")
                        Text(
                            "Gemma may read the computer's current date, time, UTC offset, and timezone.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                androidx.compose.material3.HorizontalDivider(Modifier.padding(vertical = StudioCompactGap))
                Text("Appearance", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text("Dark mode", fontWeight = FontWeight.Medium)
                        Text(
                            "Use a neutral dark surface with gem16 green accents.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(
                        checked = state.settings.darkTheme,
                        onCheckedChange = { state.toggleTheme() },
                    )
                }
            }
        }
        StudioSurface(Modifier.weight(1f)) {
            Column(
                Modifier.fillMaxWidth().padding(StudioPanelPadding),
                verticalArrangement = Arrangement.spacedBy(StudioGap),
            ) {
                Text("About gem16", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(
                    "A local, cross-platform Compose Desktop client for the qualified Gemma 4 12B path and " +
                        "the qualified text-only Gemma 4 26B A4B inference path.",
                )
                Text(
                    "The UI communicates only with gem16's OpenAI-compatible HTTP API. Model weights and prompts " +
                        "remain on the configured machine.",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Row {
                    Text("Endpoint", fontWeight = FontWeight.SemiBold)
                    Spacer(Modifier.width(12.dp))
                    Text(state.settings.server.baseUrl)
                }
            }
        }
    }
}
