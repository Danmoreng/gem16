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
    inversePrimary = Color(0xFF126C3D),
    secondary = Color(0xFFB8B8B8),
    onSecondary = Color(0xFF242424),
    secondaryContainer = Color(0xFF3A3A3A),
    onSecondaryContainer = Color(0xFFE2E2E2),
    tertiary = Color(0xFFB8B8B8),
    onTertiary = Color(0xFF242424),
    tertiaryContainer = Color(0xFF3A3A3A),
    onTertiaryContainer = Color(0xFFE2E2E2),
    background = Color(0xFF1E1E1E),
    onBackground = Color(0xFFE6E6E6),
    surface = Color(0xFF282828),
    onSurface = Color(0xFFE6E6E6),
    surfaceVariant = Color(0xFF323232),
    surfaceTint = Color(0xFF80E8A4),
    surfaceDim = Color(0xFF1A1A1A),
    surfaceBright = Color(0xFF3A3A3A),
    surfaceContainerLowest = Color(0xFF161616),
    surfaceContainerLow = Color(0xFF222222),
    surfaceContainer = Color(0xFF282828),
    surfaceContainerHigh = Color(0xFF2E2E2E),
    surfaceContainerHighest = Color(0xFF343434),
    onSurfaceVariant = Color(0xFFBDBDBD),
    inverseSurface = Color(0xFFE6E6E6),
    inverseOnSurface = Color(0xFF2A2A2A),
    outline = Color(0xFF5C5C5C),
    outlineVariant = Color(0xFF3E3E3E),
    scrim = Color.Black,
    primaryFixed = Color(0xFFB9F6CA),
    primaryFixedDim = Color(0xFF80E8A4),
    onPrimaryFixed = Color(0xFF00391E),
    onPrimaryFixedVariant = Color(0xFF126C3D),
    secondaryFixed = Color(0xFFE2E2E2),
    secondaryFixedDim = Color(0xFFB8B8B8),
    onSecondaryFixed = Color(0xFF242424),
    onSecondaryFixedVariant = Color(0xFF4A4A4A),
    tertiaryFixed = Color(0xFFE2E2E2),
    tertiaryFixedDim = Color(0xFFB8B8B8),
    onTertiaryFixed = Color(0xFF242424),
    onTertiaryFixedVariant = Color(0xFF4A4A4A),
    error = Color(0xFFFFB4AB),
)

private val GemLight = lightColorScheme(
    primary = Color(0xFF126C3D),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFB9F6CA),
    onPrimaryContainer = Color(0xFF00391E),
    inversePrimary = Color(0xFF80E8A4),
    secondary = Color(0xFF5F5F5F),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFE2E2E2),
    onSecondaryContainer = Color(0xFF292929),
    tertiary = Color(0xFF5F5F5F),
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFE2E2E2),
    onTertiaryContainer = Color(0xFF292929),
    background = Color(0xFFF3F3F3),
    onBackground = Color(0xFF202020),
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF202020),
    surfaceVariant = Color(0xFFEAEAEA),
    surfaceTint = Color(0xFF126C3D),
    surfaceDim = Color(0xFFDADADA),
    surfaceBright = Color(0xFFFFFFFF),
    surfaceContainerLowest = Color(0xFFFFFFFF),
    surfaceContainerLow = Color(0xFFF7F7F7),
    surfaceContainer = Color(0xFFF1F1F1),
    surfaceContainerHigh = Color(0xFFEBEBEB),
    surfaceContainerHighest = Color(0xFFE5E5E5),
    onSurfaceVariant = Color(0xFF575757),
    inverseSurface = Color(0xFF303030),
    inverseOnSurface = Color(0xFFF2F2F2),
    outline = Color(0xFF818181),
    outlineVariant = Color(0xFFD0D0D0),
    scrim = Color.Black,
    primaryFixed = Color(0xFFB9F6CA),
    primaryFixedDim = Color(0xFF80E8A4),
    onPrimaryFixed = Color(0xFF00391E),
    onPrimaryFixedVariant = Color(0xFF126C3D),
    secondaryFixed = Color(0xFFE2E2E2),
    secondaryFixedDim = Color(0xFFB8B8B8),
    onSecondaryFixed = Color(0xFF242424),
    onSecondaryFixedVariant = Color(0xFF4A4A4A),
    tertiaryFixed = Color(0xFFE2E2E2),
    tertiaryFixedDim = Color(0xFFB8B8B8),
    onTertiaryFixed = Color(0xFF242424),
    onTertiaryFixedVariant = Color(0xFF4A4A4A),
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
