package com.gem16.studio

import androidx.compose.runtime.remember
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import com.gem16.studio.state.StudioState
import java.awt.Dimension

fun main() = application {
    val studio = remember { StudioState() }
    val windowState = rememberWindowState(width = 1360.dp, height = 900.dp)
    Window(
        onCloseRequest = {
            studio.close()
            exitApplication()
        },
        title = "gem16",
        icon = painterResource("icons/gem16-studio.svg"),
        state = windowState,
    ) {
        window.minimumSize = Dimension(1040, 700)
        StudioApp(studio)
    }
}
