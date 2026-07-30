package com.gem16.studio.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val GemDark = darkColorScheme(
    primary = Color(0xFF80E8A4),
    onPrimary = Color(0xFF00391E),
    primaryContainer = Color(0xFF174C31),
    onPrimaryContainer = Color(0xFFB5F5C9),
    secondary = Color(0xFFAAB5AE),
    onSecondary = Color(0xFF202522),
    secondaryContainer = Color(0xFF343B37),
    onSecondaryContainer = Color(0xFFDDE5E0),
    background = Color(0xFF0D0F0E),
    onBackground = Color(0xFFE5E8E6),
    surface = Color(0xFF151817),
    onSurface = Color(0xFFE5E8E6),
    surfaceVariant = Color(0xFF222624),
    onSurfaceVariant = Color(0xFFB8BFBB),
    outline = Color(0xFF505753),
    outlineVariant = Color(0xFF303532),
    error = Color(0xFFFFB4AB),
)

private val GemLight = lightColorScheme(
    primary = Color(0xFF126C3D),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFB9F6CA),
    onPrimaryContainer = Color(0xFF00391E),
    secondary = Color(0xFF3E6650),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFD5EBDD),
    onSecondaryContainer = Color(0xFF173929),
    background = Color(0xFFF4FAF5),
    onBackground = Color(0xFF142018),
    surface = Color(0xFFFCFFFC),
    onSurface = Color(0xFF142018),
    surfaceVariant = Color(0xFFE5F0E8),
    onSurfaceVariant = Color(0xFF46534A),
    outline = Color(0xFF77867B),
    outlineVariant = Color(0xFFC7D8CB),
)

@Composable
fun Gem16Theme(dark: Boolean, content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (dark) GemDark else GemLight, content = content)
}
