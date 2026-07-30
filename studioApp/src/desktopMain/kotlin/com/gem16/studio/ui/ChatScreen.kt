package com.gem16.studio.ui

import androidx.compose.foundation.Image
import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.draganddrop.dragAndDropTarget
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
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollbarAdapter
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.ArrowDownward
import androidx.compose.material.icons.filled.AttachFile
import androidx.compose.material.icons.filled.Audiotrack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.DeleteSweep
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.draganddrop.DragAndDropEvent
import androidx.compose.ui.draganddrop.DragAndDropTarget
import androidx.compose.ui.draganddrop.DragData
import androidx.compose.ui.draganddrop.dragData
import androidx.compose.ui.graphics.toComposeImageBitmap
import androidx.compose.ui.input.key.Key
import androidx.compose.ui.input.key.KeyEventType
import androidx.compose.ui.input.key.isShiftPressed
import androidx.compose.ui.input.key.key
import androidx.compose.ui.input.key.onPreviewKeyEvent
import androidx.compose.ui.input.key.type
import androidx.compose.ui.input.pointer.PointerEventType
import androidx.compose.ui.input.pointer.onPointerEvent
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.gem16.studio.model.ChatMessage
import com.gem16.studio.model.MediaAttachment
import com.gem16.studio.model.MediaKind
import com.gem16.studio.model.ThinkingEffort
import com.gem16.studio.service.formatBytes
import com.gem16.studio.state.StudioState
import java.net.URI
import java.nio.file.Files
import java.nio.file.Path
import javax.swing.JFileChooser
import javax.swing.filechooser.FileNameExtensionFilter
import org.jetbrains.jewel.ui.component.IconButton
import org.jetbrains.jewel.ui.component.OutlinedButton
import org.jetbrains.jewel.ui.component.OutlinedSlimButton as TextButton
import org.jetbrains.jewel.ui.component.Text

@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun ChatScreen(state: StudioState) {
    val listState = rememberLazyListState()
    var autoFollow by remember { mutableStateOf(true) }
    var programmaticScroll by remember { mutableStateOf(false) }
    var dragActive by remember { mutableStateOf(false) }
    val dropTarget = remember(state) {
        object : DragAndDropTarget {
            override fun onEntered(event: DragAndDropEvent) {
                dragActive = true
            }

            override fun onExited(event: DragAndDropEvent) {
                dragActive = false
            }

            override fun onEnded(event: DragAndDropEvent) {
                dragActive = false
            }

            override fun onDrop(event: DragAndDropEvent): Boolean {
                dragActive = false
                if (state.isGenerating || state.isLoadingAttachments || state.isRecording) return false
                val paths = droppedFilePaths(event)
                if (paths.isEmpty()) return false
                state.addAttachments(paths)
                return true
            }
        }
    }
    val contentFingerprint = state.messages.sumOf { it.content.length + it.reasoning.length }

    LaunchedEffect(listState) {
        snapshotFlow {
            Triple(listState.isScrollInProgress, listState.canScrollForward, programmaticScroll)
        }.collect { (scrolling, canScrollForward, programmatic) ->
            autoFollow = nextAutoFollowState(
                current = autoFollow,
                scrollInProgress = scrolling,
                canScrollForward = canScrollForward,
                programmaticScroll = programmatic,
            )
        }
    }
    LaunchedEffect(state.messages.isEmpty()) {
        if (state.messages.isEmpty()) autoFollow = true
    }
    LaunchedEffect(state.messages.size, contentFingerprint, autoFollow) {
        if (!autoFollow || state.messages.isEmpty()) return@LaunchedEffect
        withFrameNanos { }
        val lastItem = listState.layoutInfo.totalItemsCount - 1
        if (lastItem >= 0) {
            programmaticScroll = true
            try {
                listState.scrollToItem(lastItem)
            } finally {
                programmaticScroll = false
            }
        }
    }

    val canAcceptDrop = !state.isGenerating && !state.isLoadingAttachments && !state.isRecording
    Box(
        Modifier.fillMaxSize().dragAndDropTarget(
            shouldStartDragAndDrop = { event ->
                canAcceptDrop && event.dragData() is DragData.FilesList
            },
            target = dropTarget,
        ),
    ) {
        Column(Modifier.fillMaxSize()) {
            Box(Modifier.weight(1f).fillMaxWidth()) {
                LazyColumn(
                    state = listState,
                    modifier = Modifier.fillMaxSize()
                        .onPointerEvent(PointerEventType.Scroll) {
                            if (listState.canScrollForward) autoFollow = false
                        }
                        .padding(horizontal = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(StudioGap),
                ) {
                    item { Spacer(Modifier.height(StudioGap)) }
                    if (state.messages.isEmpty()) {
                        item { WelcomeCard() }
                    }
                    items(state.messages, key = ChatMessage::id) { message ->
                        MessageCard(message, state.settings.generation.showReasoning)
                    }
                    item { Spacer(Modifier.height(8.dp)) }
                }
                VerticalScrollbar(
                    adapter = rememberScrollbarAdapter(listState),
                    modifier = Modifier.align(Alignment.CenterEnd)
                        .fillMaxHeight()
                        .onPointerEvent(PointerEventType.Press) { autoFollow = false }
                        .onPointerEvent(PointerEventType.Release) {
                            if (!listState.canScrollForward) autoFollow = true
                        },
                )
                if (!autoFollow && state.messages.isNotEmpty()) {
                    StudioPrimaryButton(
                        onClick = { autoFollow = true },
                        modifier = Modifier.align(Alignment.BottomCenter)
                            .padding(bottom = StudioGap)
                            .height(StudioControlHeight),
                    ) {
                        Icon(
                            Icons.Default.ArrowDownward,
                            contentDescription = null,
                            modifier = Modifier.size(StudioIconSize),
                        )
                        Spacer(Modifier.width(StudioGap))
                        Text("Jump to latest", color = MaterialTheme.colorScheme.onPrimary)
                    }
                }
            }
            state.chatError?.let { error ->
                Text(
                    error,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = StudioCompactGap),
                )
            }
            HorizontalDivider()
            Composer(state)
        }
        if (dragActive) DropFilesOverlay()
    }
}

@Composable
private fun DropFilesOverlay() {
    Box(
        modifier = Modifier.fillMaxSize().padding(12.dp)
            .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.96f), MaterialTheme.shapes.large)
            .border(2.dp, MaterialTheme.colorScheme.primary, MaterialTheme.shapes.large),
        contentAlignment = Alignment.Center,
    ) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(
                Icons.Default.AttachFile,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(32.dp),
            )
            Spacer(Modifier.height(StudioGap))
            Text("Drop files to attach", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text(
                "PNG, JPEG, BMP, WAV, FLAC, or MP3",
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                style = MaterialTheme.typography.bodySmall,
            )
        }
    }
}

@Composable
private fun WelcomeCard() {
    StudioSurface(
        modifier = Modifier.fillMaxWidth().padding(top = 18.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
    ) {
        Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(StudioGap)) {
            Text("Chat locally with Gemma 4", style = MaterialTheme.typography.headlineSmall)
            Text(
                "The managed gem16 server starts automatically. Send text, images, or audio; " +
                    "the model, KV cache, and optional MTP assistant remain resident on your GPU.",
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
        StudioSurface(
            modifier = Modifier.fillMaxWidth(if (user) 0.78f else 0.92f),
            color = if (user) MaterialTheme.colorScheme.primaryContainer
            else MaterialTheme.colorScheme.surfaceVariant,
        ) {
            Column(Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(StudioGap)) {
                Text(
                    if (user) "You" else "Gemma 4",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                )
                if (!user && showReasoning && (message.reasoning.isNotBlank() || message.streaming)) {
                    ReasoningBlock(message.reasoning, message.streaming)
                }
                if (message.attachments.isNotEmpty()) {
                    AttachmentGallery(message.attachments)
                }
                if (message.content.isNotBlank()) {
                    MarkdownText(message.content, Modifier.fillMaxWidth())
                } else if (message.streaming && !showReasoning) {
                    Text("Generating…", color = MaterialTheme.colorScheme.onSurfaceVariant)
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
    var expanded by remember { mutableStateOf(false) }
    Surface(
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.55f),
        shape = MaterialTheme.shapes.medium,
    ) {
        Column(Modifier.fillMaxWidth().padding(8.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Reasoning", style = MaterialTheme.typography.labelMedium, modifier = Modifier.weight(1f))
                if (streaming) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(12.dp),
                        strokeWidth = 1.5.dp,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    Spacer(Modifier.width(StudioGap))
                }
                IconButton(onClick = { expanded = !expanded }, modifier = Modifier.size(24.dp)) {
                    Icon(
                        if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                        contentDescription = if (expanded) "Hide reasoning" else "Show reasoning",
                        modifier = Modifier.size(StudioIconSize),
                    )
                }
            }
            if (expanded && reasoning.isNotBlank()) {
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

@Composable
private fun AttachmentGallery(
    attachments: List<MediaAttachment>,
    onRemove: ((String) -> Unit)? = null,
) {
    LazyRow(
        horizontalArrangement = Arrangement.spacedBy(StudioGap),
        modifier = Modifier.fillMaxWidth(),
    ) {
        items(attachments, key = MediaAttachment::id) { attachment ->
            Box {
                Surface(
                    color = MaterialTheme.colorScheme.surface.copy(alpha = 0.65f),
                    shape = MaterialTheme.shapes.medium,
                ) {
                    Column(
                        Modifier.widthIn(min = 116.dp, max = 168.dp).padding(StudioGap),
                        verticalArrangement = Arrangement.spacedBy(StudioCompactGap),
                    ) {
                        if (attachment.kind == MediaKind.Image) {
                            val bitmap = remember(attachment.id) {
                                runCatching {
                                    org.jetbrains.skia.Image.makeFromEncoded(attachment.bytes)
                                        .toComposeImageBitmap()
                                }.getOrNull()
                            }
                            if (bitmap != null) {
                                Image(
                                    bitmap = bitmap,
                                    contentDescription = attachment.fileName,
                                    modifier = Modifier.fillMaxWidth().height(76.dp),
                                    contentScale = ContentScale.Crop,
                                )
                            }
                        } else {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    Icons.Default.Audiotrack,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                )
                                Spacer(Modifier.width(StudioGap))
                                Text("Audio", style = MaterialTheme.typography.labelLarge)
                            }
                        }
                        Text(
                            attachment.fileName,
                            style = MaterialTheme.typography.labelMedium,
                            maxLines = 1,
                        )
                        Text(
                            listOfNotNull(
                                attachment.durationMillis?.let(::formatDuration),
                                formatBytes(attachment.byteSize.toLong()),
                            ).joinToString(" · "),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
                if (onRemove != null) {
                    IconButton(
                        onClick = { onRemove(attachment.id) },
                        modifier = Modifier.align(Alignment.TopEnd).size(24.dp),
                    ) {
                        Icon(Icons.Default.Close, contentDescription = "Remove ${attachment.fileName}")
                    }
                }
            }
        }
    }
}

@Composable
private fun RecordingBar(state: StudioState) {
    Surface(
        color = MaterialTheme.colorScheme.errorContainer,
        shape = MaterialTheme.shapes.medium,
    ) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = StudioGap),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(StudioGap),
        ) {
            Box(
                Modifier.size(10.dp).background(
                    MaterialTheme.colorScheme.error,
                    MaterialTheme.shapes.small,
                ),
            )
            Text(
                "Recording ${formatDuration(state.recordingMillis)} / 00:30",
                style = MaterialTheme.typography.labelLarge,
            )
            Box(
                Modifier.width(90.dp).height(7.dp)
                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.45f), MaterialTheme.shapes.small),
            ) {
                Box(
                    Modifier.fillMaxWidth(state.recordingLevel.coerceIn(0.02f, 1f)).height(7.dp)
                        .background(MaterialTheme.colorScheme.error, MaterialTheme.shapes.small),
                )
            }
            Spacer(Modifier.weight(1f))
            TextButton(
                onClick = state::cancelRecording,
                modifier = Modifier.height(StudioControlHeight),
            ) { Text("Cancel") }
            StudioPrimaryButton(
                onClick = state::stopRecording,
                modifier = Modifier.height(StudioControlHeight),
            ) {
                Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(StudioIconSize))
                Spacer(Modifier.width(6.dp))
                Text("Stop & attach", color = MaterialTheme.colorScheme.onPrimary)
            }
        }
    }
}

@Composable
private fun Composer(state: StudioState) {
    var thinkingMenu by remember { mutableStateOf(false) }
    Column(
        Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(StudioGap),
    ) {
        if (state.isRecording) {
            RecordingBar(state)
        }
        if (state.pendingAttachments.isNotEmpty()) {
            AttachmentGallery(state.pendingAttachments, state::removeAttachment)
        }
        PerformanceBar(state)
        StudioTextField(
            value = state.draft,
            onValueChange = { state.draft = it },
            modifier = Modifier.fillMaxWidth()
                .onPreviewKeyEvent { event ->
                    if (event.type == KeyEventType.KeyDown && event.key == Key.Enter && !event.isShiftPressed) {
                        state.sendMessage()
                        true
                    } else {
                        false
                    }
                },
            placeholder = "Message Gemma 4…",
            enabled = !state.isGenerating && !state.isRecording,
            singleLine = false,
            minLines = 2,
            maxLines = 6,
            minHeight = 68.dp,
            maxHeight = 144.dp,
            supportingText = "Enter to send · Shift+Enter for a new line · Drop files anywhere",
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(
                onClick = { state.addAttachments(chooseMediaPaths()) },
                enabled = !state.isGenerating && !state.isLoadingAttachments && !state.isRecording,
                modifier = Modifier.size(StudioControlHeight),
            ) {
                if (state.isLoadingAttachments) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(20.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.primary,
                    )
                } else {
                    Icon(Icons.Default.AttachFile, contentDescription = "Attach images or audio", modifier = Modifier.size(18.dp))
                }
            }
            IconButton(
                onClick = state::startRecording,
                enabled = !state.isGenerating && !state.isLoadingAttachments && !state.isRecording,
                modifier = Modifier.size(StudioControlHeight),
            ) {
                Icon(Icons.Default.Mic, contentDescription = "Record audio", modifier = Modifier.size(18.dp))
            }
            Spacer(Modifier.weight(1f))
            IconButton(
                onClick = state::removeLastExchange,
                enabled = !state.isGenerating && !state.isRecording && state.messages.isNotEmpty(),
                modifier = Modifier.size(StudioControlHeight),
            ) {
                Icon(Icons.AutoMirrored.Filled.Undo, contentDescription = "Remove last exchange", modifier = Modifier.size(18.dp))
            }
            IconButton(
                onClick = state::clearChat,
                enabled = !state.isGenerating && !state.isRecording && state.messages.isNotEmpty(),
                modifier = Modifier.size(StudioControlHeight),
            ) {
                Icon(Icons.Default.DeleteSweep, contentDescription = "New chat", modifier = Modifier.size(18.dp))
            }
            Spacer(Modifier.width(8.dp))
            Box {
                OutlinedButton(
                    onClick = { thinkingMenu = true },
                    enabled = !state.isGenerating && !state.isRecording,
                    modifier = Modifier.height(StudioControlHeight),
                ) {
                    Text("Thinking · ${state.settings.generation.thinking.label}")
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
            Spacer(Modifier.width(8.dp))
            if (state.isGenerating) {
                StudioPrimaryButton(
                    onClick = state::cancelGeneration,
                    modifier = Modifier.height(StudioControlHeight),
                ) {
                    Icon(Icons.Default.Stop, contentDescription = null, modifier = Modifier.size(StudioIconSize))
                    Spacer(Modifier.width(8.dp))
                    Text("Stop", color = MaterialTheme.colorScheme.onPrimary)
                }
            } else {
                StudioPrimaryButton(
                    onClick = state::sendMessage,
                    enabled = !state.isLoadingAttachments && !state.isRecording &&
                        (state.draft.isNotBlank() || state.pendingAttachments.isNotEmpty()),
                    modifier = Modifier.height(StudioControlHeight),
                ) {
                    Icon(Icons.AutoMirrored.Filled.Send, contentDescription = null, modifier = Modifier.size(StudioIconSize))
                    Spacer(Modifier.width(8.dp))
                    Text("Send", color = MaterialTheme.colorScheme.onPrimary)
                }
            }
        }
    }
}

@Composable
private fun PerformanceBar(state: StudioState) {
    val performance = state.performance
    val live = state.livePerformance
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(
            if (state.isGenerating) "Live response" else "Last response",
            style = MaterialTheme.typography.labelSmall,
            color = if (state.isGenerating) MaterialTheme.colorScheme.primary
            else MaterialTheme.colorScheme.onSurfaceVariant,
            fontWeight = FontWeight.SemiBold,
        )
        if (state.isGenerating) {
            StatText("Stream", live?.tokensPerSecond?.let { "%.1f tok/s".format(it) } ?: "warming up")
            StatText("TTFT", live?.let { "%.0f ms".format(it.firstTokenMilliseconds) } ?: "waiting")
            StatText("Tokens", live?.emittedTokens?.toString() ?: "0")
            StatText("Elapsed", live?.let { "%.1f s".format(it.elapsedMilliseconds / 1_000.0) } ?: "—")
        } else {
            StatText("Decode", performance?.let { "%.1f tok/s".format(it.decodeTokensPerSecond) } ?: "—")
            StatText("Prefill", performance?.let { "%.0f tok/s".format(it.prefillTokensPerSecond) } ?: "—")
            StatText("Prefill time", performance?.let { "%.0f ms".format(it.prefillMilliseconds) } ?: "—")
            StatText("Decode time", performance?.let { "%.0f ms".format(it.decodeMilliseconds) } ?: "—")
            state.usage?.let { usage ->
                StatText("Tokens", "${usage.promptTokens} in · ${usage.completionTokens} out")
            }
        }
        Spacer(Modifier.weight(1f))
        state.lastFinishReason?.let { reason ->
            Text(
                reason,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun StatText(label: String, value: String) {
    Text(
        "$label  $value",
        style = MaterialTheme.typography.labelSmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
    )
}

internal fun nextAutoFollowState(
    current: Boolean,
    scrollInProgress: Boolean,
    canScrollForward: Boolean,
    programmaticScroll: Boolean,
): Boolean = when {
    !canScrollForward -> true
    scrollInProgress && !programmaticScroll -> false
    else -> current
}

private fun formatDuration(milliseconds: Long): String {
    val seconds = (milliseconds / 1000L).coerceAtLeast(0L)
    return "%02d:%02d".format(seconds / 60L, seconds % 60L)
}

private fun chooseMediaPaths(): List<Path> {
    val chooser = JFileChooser().apply {
        dialogTitle = "Attach images or audio"
        fileSelectionMode = JFileChooser.FILES_ONLY
        isMultiSelectionEnabled = true
        isAcceptAllFileFilterUsed = false
        fileFilter = FileNameExtensionFilter(
            "Images and audio (PNG, JPEG, BMP, WAV, FLAC, MP3)",
            "png",
            "jpg",
            "jpeg",
            "bmp",
            "wav",
            "flac",
            "mp3",
        )
    }
    if (chooser.showOpenDialog(null) != JFileChooser.APPROVE_OPTION) return emptyList()
    val selected = chooser.selectedFiles.toList().ifEmpty { listOfNotNull(chooser.selectedFile) }
    return selected.map { it.toPath().toAbsolutePath().normalize() }
}

private fun fileUriToPath(value: String): Path? = runCatching {
    Path.of(URI.create(value)).toAbsolutePath().normalize()
}.getOrNull()

@OptIn(ExperimentalComposeUiApi::class)
private fun droppedFilePaths(event: DragAndDropEvent): List<Path> = runCatching {
    (event.dragData() as? DragData.FilesList)
        ?.readFiles()
        ?.mapNotNull(::fileUriToPath)
        ?.filter { Files.isRegularFile(it) }
        .orEmpty()
}.getOrDefault(emptyList())
