package com.gem16.studio.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
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

@Composable
fun SettingsScreen(state: StudioState) {
    Column(
        Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Card {
            Column(Modifier.fillMaxWidth().padding(20.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
                Text("Generation", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                var outputText by remember(state.settings.generation.maxOutputTokens) {
                    mutableStateOf(state.settings.generation.maxOutputTokens.toString())
                }
                OutlinedTextField(
                    value = outputText,
                    onValueChange = { candidate ->
                        if (candidate.all(Char::isDigit)) {
                            outputText = candidate
                            candidate.toLongOrNull()?.takeIf { it in 1L..262144L }?.let { value ->
                                state.updateGeneration { it.copy(maxOutputTokens = value) }
                            }
                        }
                    },
                    label = { Text("Maximum output tokens") },
                    supportingText = {
                        Text("Includes private reasoning and the visible answer; it cannot exceed remaining context.")
                    },
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
            }
        }
        Card {
            Column(Modifier.fillMaxWidth().padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
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
        Card {
            Column(Modifier.fillMaxWidth().padding(20.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("About gem16 Studio", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(
                    "A local, cross-platform Compose Desktop client for the specialized gem16 Gemma 4 12B " +
                        "FP8/NVFP4 inference engine.",
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
