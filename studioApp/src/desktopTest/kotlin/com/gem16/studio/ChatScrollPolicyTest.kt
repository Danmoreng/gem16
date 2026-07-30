package com.gem16.studio

import com.gem16.studio.ui.nextAutoFollowState
import kotlin.test.Test
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class ChatScrollPolicyTest {
    @Test
    fun contentGrowthKeepsFollowingWhenUserDidNotScroll() {
        assertTrue(
            nextAutoFollowState(
                current = true,
                scrollInProgress = false,
                canScrollForward = true,
                programmaticScroll = false,
            ),
        )
    }

    @Test
    fun explicitUserScrollPausesFollowing() {
        assertFalse(
            nextAutoFollowState(
                current = true,
                scrollInProgress = true,
                canScrollForward = true,
                programmaticScroll = false,
            ),
        )
    }

    @Test
    fun programmaticScrollDoesNotPauseFollowing() {
        assertTrue(
            nextAutoFollowState(
                current = true,
                scrollInProgress = true,
                canScrollForward = true,
                programmaticScroll = true,
            ),
        )
    }

    @Test
    fun returningToBottomResumesFollowing() {
        assertTrue(
            nextAutoFollowState(
                current = false,
                scrollInProgress = true,
                canScrollForward = false,
                programmaticScroll = false,
            ),
        )
    }

    @Test
    fun pausedFollowingStaysPausedAboveBottom() {
        assertFalse(
            nextAutoFollowState(
                current = false,
                scrollInProgress = false,
                canScrollForward = true,
                programmaticScroll = false,
            ),
        )
    }
}
