package com.gem16.studio.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val GemDark = darkColorScheme(
    primary = Color(0xFF8AB4F8),
    onPrimary = Color(0xFF062E6F),
    primaryContainer = Color(0xFF174EA6),
    onPrimaryContainer = Color(0xFFD2E3FC),
    secondary = Color(0xFF7DD3C7),
    background = Color(0xFF0B0F14),
    onBackground = Color(0xFFE6EDF3),
    surface = Color(0xFF121820),
    onSurface = Color(0xFFE6EDF3),
    surfaceVariant = Color(0xFF1B2430),
    onSurfaceVariant = Color(0xFFAAB7C4),
    outline = Color(0xFF344252),
    error = Color(0xFFFF6B6B),
)

private val GemLight = lightColorScheme(
    primary = Color(0xFF2459A9),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD8E6FF),
    onPrimaryContainer = Color(0xFF001A41),
    secondary = Color(0xFF226B62),
    background = Color(0xFFF6F8FB),
    onBackground = Color(0xFF17202A),
    surface = Color.White,
    onSurface = Color(0xFF17202A),
    surfaceVariant = Color(0xFFEAF0F6),
    onSurfaceVariant = Color(0xFF4C5A68),
    outline = Color(0xFFB6C1CC),
)

@Composable
fun Gem16Theme(dark: Boolean, content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (dark) GemDark else GemLight, content = content)
}
