package com.gem16.studio.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Button
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import org.jetbrains.jewel.ui.component.Text

@Composable
internal fun StudioSurface(
    modifier: Modifier = Modifier,
    color: androidx.compose.ui.graphics.Color = MaterialTheme.colorScheme.surface,
    content: @Composable BoxScope.() -> Unit,
) {
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(StudioPanelRadius))
            .background(color),
        content = content,
    )
}

@Composable
internal fun StudioPrimaryButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    content: @Composable RowScope.() -> Unit,
) {
    Button(
        onClick = onClick,
        modifier = modifier,
        enabled = enabled,
        shape = RoundedCornerShape(StudioControlRadius),
        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 0.dp),
        content = content,
    )
}

@Composable
internal fun StudioTextField(
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    label: String = "",
    placeholder: String = "",
    enabled: Boolean = true,
    singleLine: Boolean = true,
    minLines: Int = 1,
    maxLines: Int = if (singleLine) 1 else Int.MAX_VALUE,
    minHeight: Dp = if (singleLine) 34.dp else 70.dp,
    maxHeight: Dp = Dp.Unspecified,
    supportingText: String = "",
    isError: Boolean = false,
    visualTransformation: VisualTransformation = VisualTransformation.None,
) {
    val shape = RoundedCornerShape(StudioControlRadius)
    val borderColor = if (isError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.outlineVariant
    val textColor = if (enabled) MaterialTheme.colorScheme.onSurface else {
        MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
    }
    Column(modifier, verticalArrangement = Arrangement.spacedBy(StudioCompactGap)) {
        if (label.isNotBlank()) {
            Text(
                label,
                style = MaterialTheme.typography.labelMedium,
                color = if (isError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .then(
                    if (maxHeight == Dp.Unspecified) Modifier.heightIn(min = minHeight)
                    else Modifier.heightIn(min = minHeight, max = maxHeight),
                )
                .clip(shape)
                .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = if (enabled) 0.52f else 0.28f))
                .border(1.dp, borderColor, shape)
                .padding(horizontal = 9.dp, vertical = if (singleLine) 0.dp else 7.dp),
            contentAlignment = if (singleLine) Alignment.CenterStart else Alignment.TopStart,
        ) {
            BasicTextField(
                value = value,
                onValueChange = onValueChange,
                modifier = Modifier.fillMaxWidth(),
                enabled = enabled,
                singleLine = singleLine,
                minLines = minLines,
                maxLines = maxLines,
                visualTransformation = visualTransformation,
                textStyle = MaterialTheme.typography.bodyMedium.copy(color = textColor),
                cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                decorationBox = { innerTextField ->
                    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.CenterStart) {
                        if (value.isEmpty() && placeholder.isNotBlank()) {
                            Text(
                                placeholder,
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis,
                            )
                        }
                        innerTextField()
                    }
                },
            )
        }
        if (supportingText.isNotBlank()) {
            Text(
                supportingText,
                style = MaterialTheme.typography.labelSmall,
                color = if (isError) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}
