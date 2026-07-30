package com.gem16.studio.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.FolderOpen
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
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
import com.gem16.studio.model.ServerPhase
import com.gem16.studio.state.StudioState
import java.io.File
import javax.swing.JFileChooser

@Composable
fun ServerScreen(state: StudioState) {
    val phase by state.serverManager.phase.collectAsState()
    val health by state.serverManager.health.collectAsState()
    val logs by state.serverManager.logs.collectAsState()
    val error by state.serverManager.error.collectAsState()

    Row(
        Modifier.fillMaxSize().padding(20.dp),
        horizontalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        LazyColumn(
            modifier = Modifier.weight(0.95f).fillMaxHeight(),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            item { ServerStatusCard(phase, health, error) }
            item { ServerConfiguration(state, phase) }
        }
        LogPanel(
            logs = logs,
            onClear = state.serverManager::clearLogs,
            modifier = Modifier.weight(1.05f).fillMaxHeight(),
        )
    }
}

@Composable
private fun ServerStatusCard(
    phase: ServerPhase,
    health: com.gem16.studio.model.HealthSnapshot?,
    error: String?,
) {
    val online = phase == ServerPhase.Running || phase == ServerPhase.External
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
        Column(Modifier.fillMaxWidth().padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier.size(12.dp).background(
                        when {
                            online -> Color(0xFF4ADE80)
                            phase == ServerPhase.Error -> MaterialTheme.colorScheme.error
                            else -> Color(0xFFF59E0B)
                        },
                        MaterialTheme.shapes.small,
                    ),
                )
                Spacer(Modifier.width(10.dp))
                Text(
                    when (phase) {
                        ServerPhase.External -> "Connected to external server"
                        else -> "Server ${phase.name.lowercase()}"
                    },
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                )
            }
            health?.let {
                Text(
                    "Context ${formatCount(it.maxContextTokens)} · sessions ${it.residentSessions}/${it.sessionLimit} · " +
                        "MTP D${it.mtpDraftTokens} · " +
                        if (it.samplingEnabled) "sampled ${it.temperature}/${it.topP}/${it.topK}" else "greedy",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            error?.let { Text(it, color = MaterialTheme.colorScheme.error) }
        }
    }
}

@Composable
private fun ServerConfiguration(state: StudioState, phase: ServerPhase) {
    val config = state.settings.server
    val busy = phase == ServerPhase.Starting || phase == ServerPhase.Stopping
    Card {
        Column(Modifier.fillMaxWidth().padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Managed server", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            PathField(
                label = "gem16-server executable",
                value = config.executable,
                directory = false,
                onChange = { value -> state.updateServer { it.copy(executable = value) } },
            )
            PathField(
                label = "Model directory",
                value = config.modelDirectory,
                directory = true,
                onChange = { value -> state.updateServer { it.copy(modelDirectory = value) } },
            )
            PathField(
                label = "MTP assistant directory",
                value = config.assistantModelDirectory,
                directory = true,
                enabled = config.mtpDraftTokens != 0,
                onChange = { value -> state.updateServer { it.copy(assistantModelDirectory = value) } },
            )
            OutlinedTextField(
                value = config.modelName,
                onValueChange = { value -> state.updateServer { it.copy(modelName = value) } },
                label = { Text("Served model name") },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                OutlinedTextField(
                    value = config.host,
                    onValueChange = { value -> state.updateServer { it.copy(host = value) } },
                    label = { Text("Host") },
                    singleLine = true,
                    modifier = Modifier.weight(1f),
                )
                NumericField(
                    label = "Port",
                    value = config.port.toLong(),
                    range = 1L..65535L,
                    modifier = Modifier.weight(0.55f),
                    onValid = { value -> state.updateServer { it.copy(port = value.toInt()) } },
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                NumericField(
                    label = "Context tokens",
                    value = config.maxContextTokens,
                    range = 1L..262144L,
                    modifier = Modifier.weight(1f),
                    onValid = { value -> state.updateServer { it.copy(maxContextTokens = value) } },
                )
                NumericField(
                    label = "Sessions",
                    value = config.maxSessions.toLong(),
                    range = 1L..64L,
                    modifier = Modifier.weight(0.55f),
                    onValid = { value -> state.updateServer { it.copy(maxSessions = value.toInt()) } },
                )
            }
            Row(verticalAlignment = Alignment.CenterVertically) {
                var mtpMenu by remember { mutableStateOf(false) }
                Box {
                    Button(onClick = { mtpMenu = true }) {
                        Text(if (config.mtpDraftTokens == 0) "MTP off" else "MTP D${config.mtpDraftTokens}")
                    }
                    DropdownMenu(expanded = mtpMenu, onDismissRequest = { mtpMenu = false }) {
                        listOf(0, 1, 2, 4).forEach { drafts ->
                            DropdownMenuItem(
                                text = { Text(if (drafts == 0) "Off" else "D$drafts") },
                                onClick = {
                                    state.updateServer { it.copy(mtpDraftTokens = drafts) }
                                    mtpMenu = false
                                },
                            )
                        }
                    }
                }
                Spacer(Modifier.width(18.dp))
                LabeledCheckbox(
                    "Adaptive",
                    config.mtpAdaptive,
                    config.mtpDraftTokens != 0,
                ) { checked -> state.updateServer { it.copy(mtpAdaptive = checked) } }
                LabeledCheckbox("Greedy", config.greedy) { checked ->
                    state.updateServer { it.copy(greedy = checked) }
                }
                LabeledCheckbox("Auto-start", config.autoStart) { checked ->
                    state.updateServer { it.copy(autoStart = checked) }
                }
            }
            Text(
                if (config.host == "127.0.0.1" || config.host == "localhost") {
                    "The server is bound to this machine only."
                } else {
                    "Warning: gem16 has no built-in authentication or TLS. Use a trusted reverse proxy."
                },
                style = MaterialTheme.typography.bodySmall,
                color = if (config.host == "127.0.0.1" || config.host == "localhost") {
                    MaterialTheme.colorScheme.onSurfaceVariant
                } else {
                    MaterialTheme.colorScheme.error
                },
            )
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                when (phase) {
                    ServerPhase.Running, ServerPhase.Starting, ServerPhase.Stopping -> Button(
                        onClick = state::stopServer,
                        enabled = !busy,
                    ) {
                        Icon(Icons.Default.Stop, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Stop server")
                    }
                    ServerPhase.External -> Text(
                        "This process was not started by Studio and will not be stopped by it.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    else -> Button(onClick = state::startServer, enabled = !busy) {
                        Icon(Icons.Default.PlayArrow, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("Start server")
                    }
                }
            }
        }
    }
}

@Composable
private fun PathField(
    label: String,
    value: String,
    directory: Boolean,
    enabled: Boolean = true,
    onChange: (String) -> Unit,
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        OutlinedTextField(
            value = value,
            onValueChange = onChange,
            label = { Text(label) },
            singleLine = true,
            enabled = enabled,
            modifier = Modifier.weight(1f),
        )
        Spacer(Modifier.width(8.dp))
        IconButton(
            onClick = { choosePath(value, directory)?.let(onChange) },
            enabled = enabled,
        ) {
            Icon(Icons.Default.FolderOpen, contentDescription = "Browse")
        }
    }
}

@Composable
private fun NumericField(
    label: String,
    value: Long,
    range: LongRange,
    modifier: Modifier = Modifier,
    onValid: (Long) -> Unit,
) {
    var text by remember(value) { mutableStateOf(value.toString()) }
    val parsed = text.toLongOrNull()
    OutlinedTextField(
        value = text,
        onValueChange = { candidate ->
            if (candidate.all(Char::isDigit)) {
                text = candidate
                candidate.toLongOrNull()?.takeIf { it in range }?.let(onValid)
            }
        },
        label = { Text(label) },
        singleLine = true,
        isError = parsed == null || parsed !in range,
        modifier = modifier,
    )
}

@Composable
private fun LabeledCheckbox(
    label: String,
    checked: Boolean,
    enabled: Boolean = true,
    onChange: (Boolean) -> Unit,
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Checkbox(checked = checked, onCheckedChange = onChange, enabled = enabled)
        Text(label, color = if (enabled) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.outline)
    }
}

@Composable
private fun LogPanel(logs: List<String>, onClear: () -> Unit, modifier: Modifier = Modifier) {
    val listState = rememberLazyListState()
    LaunchedEffect(logs.size) {
        if (logs.isNotEmpty()) listState.scrollToItem(logs.lastIndex)
    }
    Card(modifier) {
        Column(Modifier.fillMaxSize()) {
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("Server log", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
                IconButton(onClick = onClear) { Icon(Icons.Default.Delete, contentDescription = "Clear logs") }
            }
            LazyColumn(
                state = listState,
                modifier = Modifier.fillMaxSize().background(Color(0xFF090D12)).padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(3.dp),
            ) {
                items(logs) { line ->
                    SelectionContainer {
                        Text(
                            line,
                            color = Color(0xFFB8C7D9),
                            style = MaterialTheme.typography.bodySmall,
                            fontFamily = FontFamily.Monospace,
                        )
                    }
                }
            }
        }
    }
}

private fun choosePath(current: String, directory: Boolean): String? {
    val chooser = JFileChooser().apply {
        dialogTitle = if (directory) "Select directory" else "Select executable"
        fileSelectionMode = if (directory) JFileChooser.DIRECTORIES_ONLY else JFileChooser.FILES_ONLY
        val currentFile = File(current)
        currentDirectory = if (currentFile.isDirectory) currentFile else currentFile.parentFile
        if (currentFile.exists()) selectedFile = currentFile
    }
    return if (chooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
        chooser.selectedFile.absoluteFile.normalize().path
    } else {
        null
    }
}

private fun formatCount(value: Long): String = when {
    value >= 1024 -> "${value / 1024}K"
    else -> value.toString()
}
