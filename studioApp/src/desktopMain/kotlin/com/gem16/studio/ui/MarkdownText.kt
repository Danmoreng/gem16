package com.gem16.studio.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import org.commonmark.node.BlockQuote
import org.commonmark.node.BulletList
import org.commonmark.node.Code
import org.commonmark.node.Emphasis
import org.commonmark.node.FencedCodeBlock
import org.commonmark.node.HardLineBreak
import org.commonmark.node.Heading
import org.commonmark.node.HtmlBlock
import org.commonmark.node.HtmlInline
import org.commonmark.node.Image
import org.commonmark.node.IndentedCodeBlock
import org.commonmark.node.Link
import org.commonmark.node.ListItem
import org.commonmark.node.Node
import org.commonmark.node.OrderedList
import org.commonmark.node.Paragraph
import org.commonmark.node.SoftLineBreak
import org.commonmark.node.StrongEmphasis
import org.commonmark.node.Text as MarkdownTextNode
import org.commonmark.node.ThematicBreak
import org.commonmark.parser.Parser

private const val LinkTag = "markdown-link"
private val markdownParser: Parser = Parser.builder().build()

internal fun parseMarkdown(markdown: String): Node = markdownParser.parse(markdown)

@Composable
fun MarkdownText(markdown: String, modifier: Modifier = Modifier) {
    val document = remember(markdown) { parseMarkdown(markdown) }
    Column(modifier, verticalArrangement = Arrangement.spacedBy(StudioGap)) {
        MarkdownChildren(document)
    }
}

@Composable
private fun MarkdownChildren(parent: Node) {
    var child = parent.firstChild
    while (child != null) {
        MarkdownBlock(child)
        child = child.next
    }
}

@Composable
private fun MarkdownBlock(node: Node) {
    when (node) {
        is Paragraph -> InlineMarkdown(node, MaterialTheme.typography.bodyLarge)
        is Heading -> InlineMarkdown(
            node,
            when (node.level) {
                1 -> MaterialTheme.typography.headlineMedium
                2 -> MaterialTheme.typography.headlineSmall
                3 -> MaterialTheme.typography.titleLarge
                else -> MaterialTheme.typography.titleMedium
            }.copy(fontWeight = FontWeight.Bold),
        )
        is FencedCodeBlock -> CodeBlock(node.literal, node.info)
        is IndentedCodeBlock -> CodeBlock(node.literal, null)
        is BulletList -> MarkdownList(node, ordered = false, start = 1)
        is OrderedList -> MarkdownList(node, ordered = true, start = node.markerStartNumber)
        is BlockQuote -> Surface(
            color = MaterialTheme.colorScheme.surface.copy(alpha = 0.45f),
            shape = MaterialTheme.shapes.small,
        ) {
            Row(Modifier.fillMaxWidth()) {
                Box(
                    Modifier.width(3.dp)
                        .height(32.dp)
                        .background(MaterialTheme.colorScheme.primary),
                )
                Column(Modifier.padding(horizontal = 8.dp, vertical = StudioGap)) {
                    MarkdownChildren(node)
                }
            }
        }
        is ThematicBreak -> HorizontalDivider()
        is HtmlBlock -> SelectionContainer { Text(node.literal, fontFamily = FontFamily.Monospace) }
        else -> MarkdownChildren(node)
    }
}

@Composable
private fun MarkdownList(node: Node, ordered: Boolean, start: Int) {
    Column(verticalArrangement = Arrangement.spacedBy(StudioCompactGap)) {
        var item = node.firstChild
        var index = start
        while (item != null) {
            if (item is ListItem) {
                Row(Modifier.fillMaxWidth()) {
                    Text(
                        if (ordered) "${index++}." else "•",
                        modifier = Modifier.width(22.dp),
                        color = MaterialTheme.colorScheme.primary,
                        fontWeight = FontWeight.SemiBold,
                    )
                    Column(Modifier.weight(1f)) { MarkdownChildren(item) }
                }
            }
            item = item.next
        }
    }
}

@Composable
private fun CodeBlock(code: String, language: String?) {
    Surface(
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.72f),
        shape = MaterialTheme.shapes.medium,
    ) {
        Column(Modifier.fillMaxWidth().padding(8.dp)) {
            language?.takeIf(String::isNotBlank)?.let {
                Text(
                    it.trim(),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(bottom = StudioCompactGap),
                )
            }
            SelectionContainer {
                Text(
                    code.trimEnd(),
                    modifier = Modifier.horizontalScroll(rememberScrollState()),
                    style = MaterialTheme.typography.bodyMedium,
                    fontFamily = FontFamily.Monospace,
                )
            }
        }
    }
}

@Composable
private fun InlineMarkdown(node: Node, style: TextStyle) {
    val linkColor = MaterialTheme.colorScheme.primary
    val codeBackground = MaterialTheme.colorScheme.surface
    val value = remember(node, linkColor, codeBackground) {
        buildAnnotatedString { appendInline(node, linkColor, codeBackground) }
    }
    val uriHandler = LocalUriHandler.current
    var layoutResult by remember { mutableStateOf<TextLayoutResult?>(null) }
    SelectionContainer {
        Text(
            value,
            style = style,
            onTextLayout = { layoutResult = it },
            modifier = Modifier.pointerInput(value) {
                detectTapGestures { position ->
                    val offset = layoutResult?.getOffsetForPosition(position) ?: return@detectTapGestures
                    value.getStringAnnotations(LinkTag, offset, offset)
                        .firstOrNull()
                        ?.item
                        ?.takeIf(::isSafeExternalUri)
                        ?.let { uri -> runCatching { uriHandler.openUri(uri) } }
                }
            },
        )
    }
}

private fun isSafeExternalUri(value: String): Boolean =
    value.startsWith("https://", ignoreCase = true) ||
        value.startsWith("http://", ignoreCase = true) ||
        value.startsWith("mailto:", ignoreCase = true)

private fun AnnotatedString.Builder.appendInline(node: Node, linkColor: Color, codeBackground: Color) {
    var child = node.firstChild
    while (child != null) {
        when (child) {
            is MarkdownTextNode -> append(child.literal)
            is SoftLineBreak -> append("\n")
            is HardLineBreak -> append("\n")
            is Code -> withStyle(
                SpanStyle(fontFamily = FontFamily.Monospace, background = codeBackground),
            ) { append(child.literal) }
            is Emphasis -> withStyle(SpanStyle(fontStyle = FontStyle.Italic)) {
                appendInline(child, linkColor, codeBackground)
            }
            is StrongEmphasis -> withStyle(SpanStyle(fontWeight = FontWeight.Bold)) {
                appendInline(child, linkColor, codeBackground)
            }
            is Link -> {
                val start = length
                withStyle(SpanStyle(color = linkColor, textDecoration = TextDecoration.Underline)) {
                    appendInline(child, linkColor, codeBackground)
                }
                addStringAnnotation(LinkTag, child.destination, start, length)
            }
            is Image -> {
                append("[Image")
                if (child.firstChild != null) {
                    append(": ")
                    appendInline(child, linkColor, codeBackground)
                }
                append("]")
            }
            is HtmlInline -> withStyle(SpanStyle(fontFamily = FontFamily.Monospace)) {
                append(child.literal)
            }
            else -> appendInline(child, linkColor, codeBackground)
        }
        child = child.next
    }
}
