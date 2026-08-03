import org.gradle.api.tasks.Sync
import org.jetbrains.compose.desktop.application.dsl.TargetFormat
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.jetbrainsCompose)
    alias(libs.plugins.compose.compiler)
    alias(libs.plugins.kotlinSerialization)
}

val studioPlatform = if (System.getProperty("os.name").contains("Windows", ignoreCase = true)) "Windows" else "Linux"
val studioServerName = if (studioPlatform == "Windows") "gem16-server.exe" else "gem16-server"
val studioServer = rootProject.layout.projectDirectory.file(
    "build/$studioPlatform/blackwell-release/bin/$studioServerName",
)
val generatedAppResources = layout.buildDirectory.dir("generated/studio-app-resources")
val prepareStudioAppResources by tasks.registering(Sync::class) {
    from(studioServer) { into("common/bin") }
    into(generatedAppResources)
    doFirst {
        check(studioServer.asFile.isFile) {
            "Build the release gem16-server before packaging gem16: ${studioServer.asFile}"
        }
    }
}

kotlin {
    jvmToolchain(21)

    jvm("desktop") {
        compilerOptions {
            jvmTarget.set(JvmTarget.JVM_21)
        }
    }

    sourceSets {
        val desktopMain by getting {
            dependencies {
                implementation(compose.desktop.currentOs)
                implementation(compose.material3)
                implementation("org.jetbrains.compose.material:material-icons-extended:1.7.3")
                implementation(libs.jewel.int.ui.standalone)
                implementation(libs.kotlinx.coroutines.core)
                implementation(libs.kotlinx.coroutines.swing)
                implementation(libs.kotlinx.serialization.json)
                implementation(libs.commonmark)
                implementation(libs.pdfbox)
            }
        }
        val desktopTest by getting {
            dependencies {
                implementation(libs.kotlin.test)
            }
        }
    }
}

compose.desktop {
    application {
        mainClass = "com.gem16.studio.MainKt"
        nativeDistributions {
            appResourcesRootDir.set(generatedAppResources)
            targetFormats(TargetFormat.Dmg, TargetFormat.Msi, TargetFormat.Deb)
            packageName = "gem16"
            packageVersion = providers.environmentVariable("APP_VERSION").orElse("0.1.0").get()
            description = "Desktop chat and server manager for gem16"
            vendor = "gem16"
            windows {
                iconFile.set(project.file("src/desktopMain/resources/icons/gem16-studio.ico"))
            }
            macOS {
                infoPlist {
                    extraKeysRawXml = """
                        <key>NSMicrophoneUsageDescription</key>
                        <string>gem16 records audio only when you press the microphone button.</string>
                    """.trimIndent()
                }
            }
            modules(
                "java.base",
                "java.desktop",
                "java.logging",
                "java.net.http",
                "java.prefs",
                "jdk.crypto.ec"
            )
        }
    }
}

tasks.matching {
    it.name == "prepareAppResources" ||
        it.name == "createDistributable" ||
        it.name == "packageDistributionForCurrentOS" ||
        it.name.startsWith("packageMsi") ||
        it.name.startsWith("packageDeb") ||
        it.name.startsWith("packageDmg")
}.configureEach {
    dependsOn(prepareStudioAppResources)
}
