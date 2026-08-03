package com.gem16.studio

import com.gem16.studio.ui.parseMarkdown
import org.commonmark.node.Code
import org.commonmark.node.FencedCodeBlock
import org.commonmark.node.Heading
import org.commonmark.node.Link
import org.commonmark.node.Node
import org.commonmark.node.OrderedList
import org.commonmark.node.Text
import kotlin.test.Test
import kotlin.test.assertTrue

class MarkdownParserTest {
    @Test
    fun parsesProductMarkdownBlocksAndInlineLinks() {
        val document = parseMarkdown(
            """
            # Result

            1. first
            2. [second](https://example.com)

            ```kotlin
            println("gem16")
            ```
            """.trimIndent(),
        )
        val nodes = descendants(document)
        assertTrue(nodes.any { it is Heading })
        assertTrue(nodes.any { it is OrderedList })
        assertTrue(nodes.any { it is Link && it.destination == "https://example.com" })
        assertTrue(nodes.any { it is FencedCodeBlock && it.info == "kotlin" })
    }

    @Test
    fun rendersInlineLatexArrowsButPreservesCodeAndUnknownMath() {
        val dollar = '$'
        val document = parseMarkdown(
            "Flow: Motivation ${dollar}\\rightarrow$dollar Theory " +
                "${dollar} \\Rightarrow $dollar Result; keep ${dollar}x + 1$dollar " +
                "and `${dollar}\\rightarrow$dollar`.",
        )
        val nodes = descendants(document)
        val visibleText = nodes.filterIsInstance<Text>().joinToString("") { it.literal }
        assertTrue(visibleText.contains("Motivation → Theory ⇒ Result"))
        assertTrue(visibleText.contains("keep ${dollar}x + 1$dollar"))
        assertTrue(nodes.any { it is Code && it.literal == "${dollar}\\rightarrow$dollar" })
    }
}

private fun descendants(root: Node): List<Node> = buildList {
    fun visit(node: Node) {
        add(node)
        var child = node.firstChild
        while (child != null) {
            visit(child)
            child = child.next
        }
    }
    visit(root)
}
