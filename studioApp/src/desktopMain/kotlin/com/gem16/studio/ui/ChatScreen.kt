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
import androidx.compose.material.icons.filled.Description
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
import androidx.compose.ui.input.key.isCtrlPressed
import androidx.compose.ui.input.key.isMetaPressed
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
import com.gem16.studio.model.ChatActivity
import com.gem16.studio.model.ChatActivityPhase
import com.gem16.studio.model.MediaAttachment
import com.gem16.studio.model.MediaKind
import com.gem16.studio.model.ThinkingEffort
import com.gem16.studio.model.ToolCall
import com.gem16.studio.service.formatBytes
import com.gem16.studio.state.StudioState
import java.net.URI
import java.nio.file.Files
import java.nio.file.Path
import javax.swing.JFileChooser
import javax.swing.filechooser.FileNameExtensionFilter
import kotlinx.coroutines.delay
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
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
    val visibleMessages = state.messages.filter { it.role != "tool" }
    val contentFingerprint = visibleMessages.sumOf {
        it.content.length + it.reasoning.length + it.toolCalls.size
    }

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
                    items(visibleMessages, key = ChatMessage::id) { message ->
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
                "Text, PDF, PNG, JPEG, BMP, WAV, FLAC, or MP3",
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
                "The managed gem16 server starts automatically. Send text, documents, images, or audio; " +
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
                Row(verticalAlignment = Alignment.CenterVertically) {
                    val toolRequest = !user && message.toolCalls.isNotEmpty() && message.content.isBlank()
                    Text(
                        if (user) "You" else if (toolRequest) "Gemma 4 · tool request" else "Gemma 4",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.weight(1f),
                    )
                    if (!user && message.content.isNotEmpty()) {
                        StudioCopyButton(
                            text = message.content,
                            contentDescription = "Copy complete answer",
                            modifier = Modifier.size(24.dp),
                        )
                    }
                }
                if (!user && showReasoning && (message.reasoning.isNotBlank() || message.streaming)) {
                    ReasoningBlock(message.reasoning, message.streaming)
                }
                if (message.attachments.isNotEmpty()) {
                    AttachmentGallery(message.attachments)
                }
                if (message.toolCalls.isNotEmpty()) {
                    ToolCallsBlock(message.toolCalls)
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
                        } else if (attachment.kind == MediaKind.Audio) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    Icons.Default.Audiotrack,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                )
                                Spacer(Modifier.width(StudioGap))
                                Text("Audio", style = MaterialTheme.typography.labelLarge)
                            }
                        } else {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    Icons.Default.Description,
                                    contentDescription = null,
                                    tint = MaterialTheme.colorScheme.primary,
                                )
                                Spacer(Modifier.width(StudioGap))
                                Text(
                                    if (attachment.format == "pdf") "PDF" else "Text",
                                    style = MaterialTheme.typography.labelLarge,
                                )
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
                                attachment.pageCount?.let { "$it ${if (it == 1) "page" else "pages"}" },
                                formatBytes(attachment.byteSize),
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
        state.chatActivity?.let { ActivityBar(it) }
        ContextUsageBar(state)
        PerformanceBar(state)
        StudioTextField(
            value = state.draft,
            onValueChange = { state.draft = it },
            modifier = Modifier.fillMaxWidth()
                .onPreviewKeyEvent { event ->
                    if (
                        event.type == KeyEventType.KeyDown &&
                        event.key == Key.V &&
                        (event.isCtrlPressed || event.isMetaPressed) &&
                        state.addClipboardImage()
                    ) {
                        true
                    } else if (
                        event.type == KeyEventType.KeyDown &&
                        event.key == Key.Enter &&
                        !event.isShiftPressed
                    ) {
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
            supportingText = "Enter to send · Shift+Enter for a new line · Drop files (including text/PDF) or paste images",
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
                    Icon(Icons.Default.AttachFile, contentDescription = "Attach files", modifier = Modifier.size(18.dp))
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
private fun ActivityBar(activity: ChatActivity) {
    var nowNanos by remember(activity) { mutableStateOf(System.nanoTime()) }
    LaunchedEffect(activity) {
        while (true) {
            nowNanos = System.nanoTime()
            delay(250)
        }
    }
    val elapsedMillis = ((nowNanos - activity.startedNanos) / 1_000_000L).coerceAtLeast(0L)
    val waitingLong = activity.phase == ChatActivityPhase.WaitingForFirstToken && elapsedMillis >= 30_000L
    Surface(
        color = if (waitingLong) MaterialTheme.colorScheme.errorContainer
        else MaterialTheme.colorScheme.primaryContainer,
        shape = MaterialTheme.shapes.medium,
    ) {
        Column(
            Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(StudioCompactGap),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(StudioGap),
            ) {
                CircularProgressIndicator(
                    modifier = Modifier.size(16.dp),
                    strokeWidth = 2.dp,
                    color = if (waitingLong) MaterialTheme.colorScheme.error
                    else MaterialTheme.colorScheme.primary,
                )
                Text(activity.detail, style = MaterialTheme.typography.labelLarge, modifier = Modifier.weight(1f))
                Text(
                    formatActivityDuration(elapsedMillis),
                    style = MaterialTheme.typography.labelMedium,
                    fontFamily = FontFamily.Monospace,
                )
            }
            if (waitingLong) {
                Text(
                    "No first token for 30 seconds. The stream is still open; check GPU activity and the Server logs.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                )
            }
        }
    }
}

@Composable
private fun ToolCallsBlock(toolCalls: List<ToolCall>) {
    var expanded by remember(toolCalls.map { it.id }) { mutableStateOf(false) }
    Surface(
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.55f),
        shape = MaterialTheme.shapes.medium,
    ) {
        Column(Modifier.fillMaxWidth().padding(8.dp), verticalArrangement = Arrangement.spacedBy(StudioGap)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Description, contentDescription = null, modifier = Modifier.size(StudioIconSize))
                Spacer(Modifier.width(StudioGap))
                Text(
                    if (toolCalls.size == 1) {
                        "Tool call · ${toolDisplayName(toolCalls.single().name)}"
                    } else {
                        "Tool calls · ${toolCalls.size}"
                    },
                    style = MaterialTheme.typography.labelMedium,
                    modifier = Modifier.weight(1f),
                )
                IconButton(onClick = { expanded = !expanded }, modifier = Modifier.size(24.dp)) {
                    Icon(
                        if (expanded) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                        contentDescription = if (expanded) "Hide tool call" else "Show tool call",
                        modifier = Modifier.size(StudioIconSize),
                    )
                }
            }
            if (expanded) {
                toolCalls.forEachIndexed { index, call ->
                    if (index > 0) HorizontalDivider()
                    Text(call.name, style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
                    ToolPayload("Arguments sent by Gemma", call.argumentsJson.ifBlank { "{}" })
                    ToolPayload("Result returned by the app", call.resultJson ?: "Waiting for result…")
                }
            }
        }
    }
}

@Composable
private fun ToolPayload(label: String, payload: String) {
    Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
    SelectionContainer {
        Text(
            prettyJson(payload),
            modifier = Modifier.fillMaxWidth()
                .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.65f), MaterialTheme.shapes.small)
                .padding(8.dp),
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
        )
    }
}

private fun toolDisplayName(name: String): String = when (name) {
    "get_current_date" -> "current date"
    "get_current_time" -> "current time"
    else -> name
}

private val prettyJsonEncoder = Json { prettyPrint = true }

private fun prettyJson(value: String): String = runCatching {
    val element = prettyJsonEncoder.parseToJsonElement(value)
    prettyJsonEncoder.encodeToString(JsonElement.serializer(), element)
}.getOrDefault(value)

@Composable
private fun ContextUsageBar(state: StudioState) {
    val used = state.contextTokensUsed
    val limit = state.serverManager.health.value?.maxContextTokens
        ?.takeIf { it > 0L }
        ?: state.settings.server.maxContextTokens
    val fraction = if (limit > 0L) {
        (used.toDouble() / limit.toDouble()).coerceIn(0.0, 1.0)
    } else {
        0.0
    }
    Row(
        modifier = Modifier.fillMaxWidth().padding(horizontal = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(
            "Context",
            style = MaterialTheme.typography.labelSmall,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Box(
            Modifier.weight(1f).height(6.dp)
                .background(MaterialTheme.colorScheme.surfaceVariant, MaterialTheme.shapes.small),
        ) {
            Box(
                Modifier.fillMaxWidth(fraction.toFloat()).height(6.dp)
                    .background(MaterialTheme.colorScheme.primary, MaterialTheme.shapes.small),
            )
        }
        Text(
            if (used > 0L) {
                "%,d / %,d tokens · %.1f%%".format(used, limit, fraction * 100.0)
            } else {
                "Not measured / %,d tokens".format(limit)
            },
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
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
        dialogTitle = "Attach files"
        fileSelectionMode = JFileChooser.FILES_ONLY
        isMultiSelectionEnabled = true
        isAcceptAllFileFilterUsed = false
        fileFilter = FileNameExtensionFilter(
            "Documents, images, and audio",
            "txt",
            "md",
            "markdown",
            "csv",
            "tsv",
            "json",
            "jsonl",
            "xml",
            "yaml",
            "yml",
            "log",
            "pdf",
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

private fun formatActivityDuration(milliseconds: Long): String {
    val totalSeconds = (milliseconds / 1_000L).coerceAtLeast(0L)
    return "%02d:%02d".format(totalSeconds / 60L, totalSeconds % 60L)
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
