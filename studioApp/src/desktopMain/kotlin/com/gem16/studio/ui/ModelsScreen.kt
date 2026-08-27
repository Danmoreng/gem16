package com.gem16.studio.ui

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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import com.gem16.studio.model.Gem16ModelCatalog
import com.gem16.studio.model.Gem16Qualified26BModelCatalog
import com.gem16.studio.model.ModelProfile
import com.gem16.studio.model.ServerPhase
import com.gem16.studio.service.ModelInstallState
import com.gem16.studio.service.formatBytes
import com.gem16.studio.state.StudioState
import org.jetbrains.jewel.ui.component.OutlinedButton
import org.jetbrains.jewel.ui.component.Text

@Composable
fun ModelsScreen(state: StudioState) {
    val install by state.modelManager.state.collectAsState()
    val serverPhase by state.serverManager.phase.collectAsState()
    var token by remember { mutableStateOf("") }
    val configured12B = state.settings.server.modelProfile == ModelProfile.Gemma4Unified12B &&
        state.settings.server.modelDirectory == install.targetDirectory.toString() &&
        state.settings.server.assistantModelDirectory == install.assistantDirectory.toString()
    val configured26B = state.settings.server.modelProfile == ModelProfile.Gemma4Moe26BA4B &&
        state.settings.server.modelDirectory == install.target26BDirectory.toString() &&
        state.settings.server.assistantModelDirectory == install.assistant26BDirectory.toString()

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(StudioScreenPadding),
        verticalArrangement = Arrangement.spacedBy(StudioGap),
    ) {
        StudioSurface(Modifier.fillMaxWidth()) {
            Column(
                Modifier.fillMaxWidth().padding(StudioPanelPadding),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                Text("Models", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                Text(
                    "Download the pinned gem16 model set directly into the shared Hugging Face cache. " +
                        "Existing Hub files are reused and model payloads are never copied into the application.",
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                ModelComponentRow(
                    title = "Gemma 4 12B NVFP4",
                    detail = "${Gem16ModelCatalog.targetRepository} · ${formatBytes(Gem16ModelCatalog.target.totalBytes)}",
                    ready = install.targetReady,
                )
                ModelComponentRow(
                    title = "MTP draft assistant",
                    detail = "${Gem16ModelCatalog.assistantRepository} · ${formatBytes(Gem16ModelCatalog.assistant.totalBytes)}",
                    ready = install.assistantReady,
                )
                ModelComponentRow(
                    title = "Official tokenizer configuration",
                    detail = "${Gem16ModelCatalog.tokenizerRepository}/tokenizer_config.json",
                    ready = install.tokenizerConfigReady,
                )
                Spacer(Modifier.height(StudioCompactGap))
                Text("Qualified Gemma 4 26B A4B", fontWeight = FontWeight.SemiBold)
                ModelComponentRow(
                    title = "GEM16 SM120 Target",
                    detail = "${Gem16Qualified26BModelCatalog.targetRepository} · " +
                        formatBytes(Gem16Qualified26BModelCatalog.target.totalBytes),
                    ready = install.target26BReady,
                )
                ModelComponentRow(
                    title = "GEM16 fixed-D2 Assistant",
                    detail = "${Gem16Qualified26BModelCatalog.assistantRepository} · " +
                        formatBytes(Gem16Qualified26BModelCatalog.assistant.totalBytes),
                    ready = install.assistant26BReady,
                )
                Text(
                    "Text-only · one session · fixed D2 · 73,728-token qualified MTP context",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }

        StudioSurface(Modifier.fillMaxWidth()) {
            Column(
                Modifier.fillMaxWidth().padding(StudioPanelPadding),
                verticalArrangement = Arrangement.spacedBy(StudioGap),
            ) {
                Text("Hugging Face cache", style = MaterialTheme.typography.titleMedium)
                Text(
                    install.cacheRoot.toString(),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                StudioTextField(
                    value = token,
                    onValueChange = { token = it },
                    label = "Access token (optional)",
                    placeholder = "Uses HF_TOKEN or your existing Hugging Face login when empty",
                    singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                    supportingText = "The token is kept in memory only. Gated Google repositories require accepted licenses.",
                )

                if (install.isDownloading) {
                    DownloadProgress(install)
                }
                install.error?.let {
                    Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                }

                if (install.isDownloading) {
                    OutlinedButton(
                        onClick = state::cancelModelDownload,
                        modifier = Modifier.height(StudioControlHeight),
                    ) {
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(StudioGap),
                        ) {
                            Icon(
                                Icons.Default.Pause,
                                contentDescription = null,
                                modifier = Modifier.size(StudioIconSize),
                            )
                            Text("Pause")
                        }
                    }
                } else {
                    Column(verticalArrangement = Arrangement.spacedBy(StudioGap)) {
                        Text("Gemma 4 12B Unified", fontWeight = FontWeight.Medium)
                        when {
                            !install.allReady -> StudioPrimaryButton(
                                onClick = { state.downloadModels(token.takeIf(String::isNotBlank)) },
                                modifier = Modifier.height(StudioControlHeight),
                            ) {
                                Icon(
                                    Icons.Default.Download,
                                    contentDescription = null,
                                    modifier = Modifier.size(StudioIconSize),
                                )
                                Spacer(Modifier.width(StudioGap))
                                Text(
                                    "Download ${formatBytes(Gem16ModelCatalog.totalBytes)}",
                                    color = MaterialTheme.colorScheme.onPrimary,
                                )
                            }
                            !configured12B -> StudioPrimaryButton(
                                onClick = state::useCachedModels,
                                modifier = Modifier.height(StudioControlHeight),
                            ) {
                                Icon(
                                    Icons.Default.Check,
                                    contentDescription = null,
                                    modifier = Modifier.size(StudioIconSize),
                                )
                                Spacer(Modifier.width(StudioGap))
                                Text("Use cached 12B", color = MaterialTheme.colorScheme.onPrimary)
                            }
                            else -> Text(
                                "12B is configured.",
                                color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.Medium,
                            )
                        }

                        Text("Gemma 4 26B A4B", fontWeight = FontWeight.Medium)
                        when {
                            !install.all26BReady -> StudioPrimaryButton(
                                onClick = { state.download26BModels(token.takeIf(String::isNotBlank)) },
                                modifier = Modifier.height(StudioControlHeight),
                            ) {
                                Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(StudioIconSize))
                                Spacer(Modifier.width(StudioGap))
                                Text(
                                    "Download ${formatBytes(Gem16Qualified26BModelCatalog.totalBytes)}",
                                    color = MaterialTheme.colorScheme.onPrimary,
                                )
                            }
                            !configured26B -> StudioPrimaryButton(
                                onClick = state::useCached26BModels,
                                modifier = Modifier.height(StudioControlHeight),
                            ) {
                                Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.size(StudioIconSize))
                                Spacer(Modifier.width(StudioGap))
                                Text("Use cached 26B", color = MaterialTheme.colorScheme.onPrimary)
                            }
                            else -> Text(
                                "26B is configured.",
                                color = MaterialTheme.colorScheme.primary,
                                fontWeight = FontWeight.Medium,
                            )
                        }

                        if ((configured12B || configured26B) &&
                            (serverPhase == ServerPhase.Stopped || serverPhase == ServerPhase.Error)
                        ) {
                            StudioPrimaryButton(
                                onClick = state::startServer,
                                modifier = Modifier.height(StudioControlHeight),
                            ) {
                                Icon(
                                    Icons.Default.PlayArrow,
                                    contentDescription = null,
                                    modifier = Modifier.size(StudioIconSize),
                                )
                                Spacer(Modifier.width(StudioGap))
                                Text("Start server", color = MaterialTheme.colorScheme.onPrimary)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ModelComponentRow(title: String, detail: String, ready: Boolean) {
    Row(
        Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Box(
            Modifier.size(9.dp).background(
                if (ready) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline,
                CircleShape,
            ),
        )
        Column(Modifier.weight(1f)) {
            Text(title, fontWeight = FontWeight.Medium)
            Text(detail, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Text(
            if (ready) "Cached" else "Missing",
            style = MaterialTheme.typography.labelMedium,
            color = if (ready) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun DownloadProgress(install: ModelInstallState) {
    Column(verticalArrangement = Arrangement.spacedBy(StudioCompactGap)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(
                install.currentFile ?: "Preparing cache…",
                modifier = Modifier.weight(1f),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
            )
            Text(
                "${formatBytes(install.downloadedBytes)} / ${formatBytes(install.totalBytes)}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Box(
            Modifier.fillMaxWidth().height(5.dp)
                .background(MaterialTheme.colorScheme.surfaceVariant, CircleShape),
        ) {
            Box(
                Modifier.fillMaxWidth(install.progress).height(5.dp)
                    .background(MaterialTheme.colorScheme.primary, CircleShape),
            )
        }
    }
}
