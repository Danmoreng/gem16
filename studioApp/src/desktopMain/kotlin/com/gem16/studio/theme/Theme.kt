package com.gem16.studio.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.jetbrains.jewel.intui.standalone.theme.IntUiTheme

private val GemDark = darkColorScheme(
    primary = Color(0xFF80E8A4),
    onPrimary = Color(0xFF00391E),
    primaryContainer = Color(0xFF174C31),
    onPrimaryContainer = Color(0xFFB5F5C9),
    secondary = Color(0xFFAAB5AE),
    onSecondary = Color(0xFF202522),
    secondaryContainer = Color(0xFF343B37),
    onSecondaryContainer = Color(0xFFDDE5E0),
    background = Color(0xFF1E1F22),
    onBackground = Color(0xFFE5E8E6),
    surface = Color(0xFF282A2E),
    onSurface = Color(0xFFE5E8E6),
    surfaceVariant = Color(0xFF303238),
    onSurfaceVariant = Color(0xFFB8BFBB),
    outline = Color(0xFF59605C),
    outlineVariant = Color(0xFF3B3E43),
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

private val StudioTypography = Typography(
    bodyLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontSize = 14.sp, lineHeight = 20.sp),
    bodyMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontSize = 13.sp, lineHeight = 18.sp),
    bodySmall = TextStyle(fontFamily = FontFamily.SansSerif, fontSize = 12.sp, lineHeight = 16.sp),
    labelLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontSize = 12.sp,
        lineHeight = 16.sp,
        fontWeight = FontWeight.Medium,
    ),
    labelMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontSize = 11.sp, lineHeight = 14.sp),
    labelSmall = TextStyle(fontFamily = FontFamily.SansSerif, fontSize = 10.sp, lineHeight = 13.sp),
    titleLarge = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontSize = 16.sp,
        lineHeight = 21.sp,
        fontWeight = FontWeight.SemiBold,
    ),
    titleMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontSize = 14.sp,
        lineHeight = 19.sp,
        fontWeight = FontWeight.SemiBold,
    ),
    titleSmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontSize = 13.sp,
        lineHeight = 18.sp,
        fontWeight = FontWeight.SemiBold,
    ),
    headlineSmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontSize = 20.sp,
        lineHeight = 27.sp,
        fontWeight = FontWeight.SemiBold,
    ),
    headlineMedium = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontSize = 22.sp,
        lineHeight = 29.sp,
        fontWeight = FontWeight.SemiBold,
    ),
)

private val StudioShapes = Shapes(
    extraSmall = androidx.compose.foundation.shape.RoundedCornerShape(4.dp),
    small = androidx.compose.foundation.shape.RoundedCornerShape(5.dp),
    medium = androidx.compose.foundation.shape.RoundedCornerShape(10.dp),
    large = androidx.compose.foundation.shape.RoundedCornerShape(10.dp),
    extraLarge = androidx.compose.foundation.shape.RoundedCornerShape(12.dp),
)

@Composable
fun Gem16Theme(dark: Boolean, content: @Composable () -> Unit) {
    IntUiTheme(isDark = dark) {
        MaterialTheme(
            colorScheme = if (dark) GemDark else GemLight,
            typography = StudioTypography,
            shapes = StudioShapes,
            content = content,
        )
    }
}
