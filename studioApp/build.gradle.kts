import org.jetbrains.compose.desktop.application.dsl.TargetFormat
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.jetbrainsCompose)
    alias(libs.plugins.compose.compiler)
    alias(libs.plugins.kotlinSerialization)
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
                implementation(libs.kotlinx.coroutines.core)
                implementation(libs.kotlinx.coroutines.swing)
                implementation(libs.kotlinx.serialization.json)
                implementation(libs.commonmark)
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
            targetFormats(TargetFormat.Dmg, TargetFormat.Msi, TargetFormat.Deb)
            packageName = "gem16-studio"
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
                        <string>gem16 Studio records audio only when you press the microphone button.</string>
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
