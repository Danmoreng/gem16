package com.gem16.studio.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.gem16.studio.model.ChatMessage
import com.gem16.studio.model.ServerPhase
import com.gem16.studio.model.ThinkingEffort
import com.gem16.studio.state.StudioState

@Composable
fun ChatScreen(state: StudioState, onOpenServer: () -> Unit) {
    val phase by state.serverManager.phase.collectAsState()
    val listState = rememberLazyListState()
    val contentFingerprint = state.messages.sumOf { it.content.length + it.reasoning.length }
    LaunchedEffect(state.messages.size, contentFingerprint) {
        if (state.messages.isNotEmpty()) listState.animateScrollToItem(state.messages.lastIndex)
    }

    Column(Modifier.fillMaxSize()) {
        ServerBanner(phase, onOpenServer)
        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 24.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            item { Spacer(Modifier.height(8.dp)) }
            if (state.messages.isEmpty()) {
                item { WelcomeCard() }
            }
            items(state.messages, key = ChatMessage::id) { message ->
                MessageCard(message, state.settings.generation.showReasoning)
            }
            item { Spacer(Modifier.height(8.dp)) }
        }
        state.chatError?.let { error ->
            Text(
                error,
                color = MaterialTheme.colorScheme.error,
                modifier = Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 6.dp),
            )
        }
        HorizontalDivider()
        Composer(state)
    }
}

@Composable
private fun ServerBanner(phase: ServerPhase, onOpenServer: () -> Unit) {
    val online = phase == ServerPhase.Running || phase == ServerPhase.External
    Surface(color = if (online) Color(0xFF153C35) else MaterialTheme.colorScheme.surfaceVariant) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 24.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier.size(9.dp).background(
                    if (online) Color(0xFF4ADE80) else Color(0xFFF59E0B),
                    MaterialTheme.shapes.small,
                ),
            )
            Spacer(Modifier.width(10.dp))
            Text(
                if (online) "gem16 server ready" else "gem16 server is not reachable",
                style = MaterialTheme.typography.labelLarge,
                modifier = Modifier.weight(1f),
            )
            TextButton(onClick = onOpenServer) { Text(if (online) "Manage" else "Open server") }
        }
    }
}

@Composable
private fun WelcomeCard() {
    Card(
        modifier = Modifier.fillMaxWidth().padding(top = 32.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
    ) {
        Column(Modifier.padding(24.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Chat locally with Gemma 4", style = MaterialTheme.typography.headlineSmall)
            Text(
                "Start the managed gem16 server, then send a message. The model, KV cache, " +
                    "sampling, and optional MTP assistant remain resident on your GPU.",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun MessageCard(message: ChatMessage, showReasoning: Boolean) {
    val user = message.role == "user"
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = if (user) Arrangement.End else Arrangement.Start,
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(if (user) 0.78f else 0.92f),
            colors = CardDefaults.cardColors(
                containerColor = if (user) {
                    MaterialTheme.colorScheme.primaryContainer
                } else {
                    MaterialTheme.colorScheme.surfaceVariant
                },
            ),
        ) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    if (user) "You" else "Gemma 4",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                )
                if (!user && showReasoning && message.reasoning.isNotBlank()) {
                    ReasoningBlock(message.reasoning, message.streaming)
                }
                if (message.content.isNotBlank()) {
                    SelectionContainer {
                        Text(message.content, style = MaterialTheme.typography.bodyLarge)
                    }
                } else if (message.streaming) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
                        Spacer(Modifier.width(10.dp))
                        Text("Thinking…", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                message.error?.let {
                    Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

@Composable
private fun ReasoningBlock(reasoning: String, streaming: Boolean) {
    var expanded by remember { mutableStateOf(streaming) }
    Surface(
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.55f),
        shape = MaterialTheme.shapes.medium,
    ) {
        Column(Modifier.fillMaxWidth().padding(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Reasoning", style = MaterialTheme.typography.labelMedium, modifier = Modifier.weight(1f))
                IconButton(onClick = { expanded = !expanded }, modifier = Modifier.size(28.dp)) {
                    Icon(
                        if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                        contentDescription = if (expanded) "Hide reasoning" else "Show reasoning",
                    )
                }
            }
            if (expanded) {
                SelectionContainer {
                    Text(
                        reasoning,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontFamily = FontFamily.Monospace,
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun Composer(state: StudioState) {
    var thinkingMenu by remember { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth().padding(18.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        OutlinedTextField(
            value = state.draft,
            onValueChange = { state.draft = it },
            modifier = Modifier.fillMaxWidth().heightIn(min = 92.dp, max = 180.dp),
            placeholder = { Text("Message Gemma 4…") },
            enabled = !state.isGenerating,
            minLines = 3,
            maxLines = 8,
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box {
                Button(onClick = { thinkingMenu = true }, enabled = !state.isGenerating) {
                    Text("Thinking: ${state.settings.generation.thinking.label}")
                }
                DropdownMenu(expanded = thinkingMenu, onDismissRequest = { thinkingMenu = false }) {
                    ThinkingEffort.entries.forEach { effort ->
                        DropdownMenuItem(
                            text = { Text(effort.label) },
                            onClick = {
                                state.updateGeneration { it.copy(thinking = effort) }
                                thinkingMenu = false
                            },
                        )
                    }
                }
            }
            state.usage?.let { usage ->
                Spacer(Modifier.width(14.dp))
                Text(
                    "${usage.promptTokens} in · ${usage.completionTokens} out",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Spacer(Modifier.weight(1f))
            IconButton(onClick = state::removeLastExchange, enabled = !state.isGenerating && state.messages.isNotEmpty()) {
                Icon(Icons.AutoMirrored.Filled.Undo, contentDescription = "Remove last exchange")
            }
            IconButton(onClick = state::clearChat, enabled = !state.isGenerating && state.messages.isNotEmpty()) {
                Icon(Icons.Default.DeleteSweep, contentDescription = "New chat")
            }
            Spacer(Modifier.width(8.dp))
            if (state.isGenerating) {
                Button(onClick = state::cancelGeneration) {
                    Icon(Icons.Default.Stop, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Stop")
                }
            } else {
                Button(onClick = state::sendMessage, enabled = state.draft.isNotBlank()) {
                    Icon(Icons.AutoMirrored.Filled.Send, contentDescription = null)
                    Spacer(Modifier.width(8.dp))
                    Text("Send")
                }
            }
        }
    }
}
