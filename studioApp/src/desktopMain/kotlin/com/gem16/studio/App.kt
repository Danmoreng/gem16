package com.gem16.studio

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.DarkMode
import androidx.compose.material.icons.filled.LightMode
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationRail
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
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
                NavigationRail(
                    modifier = Modifier.width(86.dp),
                    containerColor = MaterialTheme.colorScheme.surface,
                    header = {
                        Box(
                            Modifier.padding(vertical = 14.dp).size(52.dp).background(
                                MaterialTheme.colorScheme.primary,
                                MaterialTheme.shapes.large,
                            ),
                            contentAlignment = Alignment.Center,
                        ) {
                            Text("G16", color = Color.White, fontWeight = FontWeight.Bold)
                        }
                    },
                ) {
                    Spacer(Modifier.height(8.dp))
                    StudioScreen.entries.forEach { target ->
                        NavigationRailItem(
                            selected = screen == target,
                            onClick = { screen = target },
                            icon = { Icon(target.icon, contentDescription = target.label) },
                            label = { Text(target.label) },
                            alwaysShowLabel = true,
                        )
                    }
                    Spacer(Modifier.weight(1f))
                    IconButton(onClick = state::toggleTheme) {
                        Icon(
                            if (state.settings.darkTheme) Icons.Default.LightMode else Icons.Default.DarkMode,
                            contentDescription = "Toggle theme",
                        )
                    }
                    Spacer(Modifier.height(12.dp))
                }
                Column(Modifier.fillMaxSize()) {
                    Header(screen)
                    Box(Modifier.fillMaxSize()) {
                        when (screen) {
                            StudioScreen.Chat -> ChatScreen(state) { screen = StudioScreen.Server }
                            StudioScreen.Server -> ServerScreen(state)
                            StudioScreen.Settings -> SettingsScreen(state)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun Header(screen: StudioScreen) {
    Surface(tonalElevation = 1.dp, color = MaterialTheme.colorScheme.surface) {
        Row(
            Modifier.fillMaxWidth().height(64.dp).padding(horizontal = 24.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Column {
                Text(screen.label, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                Text(
                    when (screen) {
                        StudioScreen.Chat -> "Local Gemma 4 12B conversation"
                        StudioScreen.Server -> "Launch, attach, and inspect gem16-server"
                        StudioScreen.Settings -> "Generation and application preferences"
                    },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
