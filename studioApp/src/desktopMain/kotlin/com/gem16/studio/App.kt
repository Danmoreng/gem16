package com.gem16.studio

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.hoverable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsHoveredAsState
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.CloudDownload
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.gem16.studio.model.ServerPhase
import com.gem16.studio.state.StudioState
import com.gem16.studio.theme.Gem16Theme
import com.gem16.studio.ui.ChatScreen
import com.gem16.studio.ui.ModelsScreen
import com.gem16.studio.ui.ServerScreen
import com.gem16.studio.ui.SettingsScreen
import com.gem16.studio.ui.StudioCompactGap
import com.gem16.studio.ui.StudioPanelRadius
import org.jetbrains.jewel.ui.component.Text

enum class StudioScreen(val label: String, val icon: ImageVector) {
    Chat("Chat", Icons.AutoMirrored.Filled.Chat),
    Models("Models", Icons.Default.CloudDownload),
    Server("Server", Icons.Default.Storage),
    Settings("Settings", Icons.Default.Settings),
}

@Composable
fun StudioApp(state: StudioState) {
    var screen by remember {
        mutableStateOf(if (state.modelManager.state.value.allReady) StudioScreen.Chat else StudioScreen.Models)
    }
    Gem16Theme(dark = state.settings.darkTheme) {
        Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Row(Modifier.fillMaxSize()) {
                StudioSidebar(
                    state = state,
                    screen = screen,
                    onScreenSelected = { screen = it },
                )
                Box(Modifier.fillMaxSize()) {
                    when (screen) {
                        StudioScreen.Chat -> ChatScreen(state)
                        StudioScreen.Models -> ModelsScreen(state)
                        StudioScreen.Server -> ServerScreen(state)
                        StudioScreen.Settings -> SettingsScreen(state)
                    }
                }
            }
        }
    }
}

@Composable
private fun StudioSidebar(
    state: StudioState,
    screen: StudioScreen,
    onScreenSelected: (StudioScreen) -> Unit,
) {
    val phase by state.serverManager.phase.collectAsState()
    Surface(
        modifier = Modifier.width(72.dp).fillMaxHeight(),
        color = MaterialTheme.colorScheme.surface,
    ) {
        Column(
            Modifier.fillMaxSize().padding(vertical = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Image(
                painter = painterResource("icons/gem16-studio.svg"),
                contentDescription = "gem16",
                modifier = Modifier.size(40.dp).padding(2.dp),
            )
            Spacer(Modifier.height(8.dp))
            StudioScreen.entries.forEach { target ->
                StudioNavigationItem(
                    target = target,
                    selected = screen == target,
                    onClick = { onScreenSelected(target) },
                    phase = if (target == StudioScreen.Server) phase else null,
                )
                Spacer(Modifier.height(StudioCompactGap))
            }

            Spacer(Modifier.weight(1f))
        }
    }
}

@Composable
private fun StudioNavigationItem(
    target: StudioScreen,
    selected: Boolean,
    onClick: () -> Unit,
    phase: ServerPhase?,
) {
    val interactionSource = remember { MutableInteractionSource() }
    val hovered by interactionSource.collectIsHoveredAsState()
    val foreground = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
    val background = when {
        selected -> MaterialTheme.colorScheme.primary.copy(alpha = 0.16f)
        hovered -> MaterialTheme.colorScheme.onSurface.copy(alpha = 0.08f)
        else -> Color.Transparent
    }
    val shape = RoundedCornerShape(StudioPanelRadius)
    Column(
        modifier = Modifier
            .width(58.dp)
            .clip(shape)
            .background(background)
            .hoverable(interactionSource)
            .clickable(
                interactionSource = interactionSource,
                indication = null,
                onClick = onClick,
            )
            .padding(vertical = 7.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box {
            Icon(target.icon, contentDescription = null, tint = foreground, modifier = Modifier.size(19.dp))
            phase?.let {
                Box(Modifier.align(Alignment.TopEnd).padding(start = 14.dp)) { ServerIndicator(it) }
            }
        }
        Spacer(Modifier.height(2.dp))
        Text(target.label, color = foreground, fontSize = 10.sp, lineHeight = 12.sp)
    }
}

@Composable
private fun ServerIndicator(phase: ServerPhase) {
    if (phase == ServerPhase.Starting || phase == ServerPhase.Stopping) {
        CircularProgressIndicator(
            modifier = Modifier.size(13.dp),
            strokeWidth = 2.dp,
            color = MaterialTheme.colorScheme.primary,
        )
        return
    }
    val color = when (phase) {
        ServerPhase.Running, ServerPhase.External -> Color(0xFF55D98A)
        ServerPhase.Starting, ServerPhase.Stopping -> Color(0xFFF0B95A)
        ServerPhase.Error -> MaterialTheme.colorScheme.error
        ServerPhase.Stopped -> MaterialTheme.colorScheme.outline
    }
    Box(Modifier.size(9.dp).background(color, CircleShape))
}
