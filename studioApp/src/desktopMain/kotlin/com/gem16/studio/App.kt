package com.gem16.studio

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.gem16.studio.model.ServerPhase
import com.gem16.studio.state.StudioState
import com.gem16.studio.theme.Gem16Theme
import com.gem16.studio.ui.ChatScreen
import com.gem16.studio.ui.ServerScreen
import com.gem16.studio.ui.SettingsScreen

enum class StudioScreen(val label: String, val icon: ImageVector) {
    Chat("Chat", Icons.AutoMirrored.Filled.Chat),
    Server("Server", Icons.Default.Storage),
    Settings("Settings", Icons.Default.Settings),
}

@Composable
fun StudioApp(state: StudioState) {
    var screen by remember { mutableStateOf(StudioScreen.Chat) }
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
        modifier = Modifier.width(224.dp).fillMaxHeight(),
        color = MaterialTheme.colorScheme.surface,
    ) {
        Column(Modifier.fillMaxSize().padding(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 6.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Image(
                    painter = painterResource("icons/gem16-studio.svg"),
                    contentDescription = "gem16 logo",
                    modifier = Modifier.size(50.dp),
                )
                Spacer(Modifier.width(12.dp))
                Column {
                    Text("gem16", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
                    Text(
                        "Studio",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }

            Spacer(Modifier.size(18.dp))
            StudioScreen.entries.forEach { target ->
                NavigationDrawerItem(
                    label = { Text(target.label) },
                    selected = screen == target,
                    onClick = { onScreenSelected(target) },
                    icon = { Icon(target.icon, contentDescription = null) },
                    badge = if (target == StudioScreen.Server) {
                        { ServerIndicator(phase) }
                    } else {
                        null
                    },
                    modifier = Modifier.padding(vertical = 2.dp),
                )
            }

            Spacer(Modifier.weight(1f))
        }
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
