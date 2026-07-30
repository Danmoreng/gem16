package com.gem16.studio.model

import java.nio.file.Path

data class HuggingFaceSource(
    val repository: String,
    val revision: String,
    val path: String,
)

data class LockedModelFile(
    val path: String,
    val size: Long,
    val sha256: String,
    val gitOid: String,
    val lfsOid: String? = null,
    val source: HuggingFaceSource? = null,
) {
    fun source(defaultRepository: String, defaultRevision: String): HuggingFaceSource =
        source ?: HuggingFaceSource(defaultRepository, defaultRevision, path)

    val blobId: String get() = lfsOid ?: gitOid
}

data class LockedModel(
    val label: String,
    val repository: String,
    val revision: String,
    val files: List<LockedModelFile>,
) {
    val totalBytes: Long get() = files.sumOf(LockedModelFile::size)
}

object Gem16ModelCatalog {
    const val targetRepository = "unsloth/gemma-4-12b-it-NVFP4"
    const val targetRevision = "b1f649734b34aa5575b03d186abd1b9be3d0d5c4"
    const val assistantRepository = "google/gemma-4-12B-it-assistant"
    const val assistantRevision = "364bd03c9952e5b7da73665ee30c9eccfc408345"
    const val tokenizerRepository = "google/gemma-4-12B-it"
    const val tokenizerRevision = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"

    val target = LockedModel(
        label = "Gemma 4 12B NVFP4",
        repository = targetRepository,
        revision = targetRevision,
        files = listOf(
            file(".gitattributes", 1_570, "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930", "52373fe24473b1aa44333d318f578ae6bf04b49b"),
            file("README.md", 29_820, "9e2712c56330ea7eefc00a10fd3c37a63364b92e52f10e12266b01a3dc6a9887", "f43dcf66ea657ff11cc683e53122b219bce2e8df"),
            file("chat_template.jinja", 18_924, "845f1ee48e39fc942fe190da9df6a1c5db229e17a96ea08966ad1c9274e73d1b", "e929d662bbbf430219c28133472e0956733a8f84"),
            file("config.json", 7_292, "bcc0ec0398a9dd0b09586f835f17c05ed2cce99d958dd59ef629ce77e618ee49", "895577e08537dbce788bdc7a9102bf85bea2e91f"),
            file("generation_config.json", 255, "801ecff5b38d5a5f5072cd1fb0ee03afabed577fb754518266fea69453acfe6b", "b052ccdb6a742edb72bc98d7d7dd02f0c58e4680"),
            file("model.safetensors", 9_304_966_064, "7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b", "65cbf46b3af2036a6146597d1bdbf925ba13b235", "7c2ee23298e7c3a9247e8947597dca5a38f8b791a0322487466d2bfad8ce704b"),
            file("processor_config.json", 1_382, "6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9", "b889adcd78015bf5a0139438c5fb0b7067529a61"),
            file("recipe.yaml", 661, "bb19ac4f3bf8c4ffb9b88701bef8faf06c679ae9461751cc849375bb1731c17b", "c27417aceaf0637177e9d44c48ea59ccef873503"),
            file("tokenizer.json", 32_169_726, "adbaa8175acf7609b4359724f40eff359ec4fac1a8647eeb99d4422be708e1cf", "03d63e59c9102f1ba39ef281fe2e24fa71d0eb1b", "adbaa8175acf7609b4359724f40eff359ec4fac1a8647eeb99d4422be708e1cf"),
            file(
                "tokenizer_config.json",
                3_089,
                "a62f4e85a47c0c136edaaa3a4f591fd6783717299a9def47e5ad03a49f6a5eb9",
                "6520cdc25462a5eba96fdac6d8cf3caa461feb35",
                source = HuggingFaceSource(tokenizerRepository, tokenizerRevision, "tokenizer_config.json"),
            ),
        ),
    )

    val assistant = LockedModel(
        label = "Gemma 4 MTP assistant",
        repository = assistantRepository,
        revision = assistantRevision,
        files = listOf(
            file(".gitattributes", 1_624, "484fac0cb8b057eefe1992c8b72ac6e7438c7d17bd60c0e278b401c2190f7e72", "602d20f6eefed4b62821062891a7a495e25435f9"),
            file("README.md", 29_899, "eb9c31937729ee1da6767900f2d3ee79e72f070fdb0181b68983df742abc97d8", "d4d487c449ba880eda5b75718a36a2b65b7ac65b"),
            file("config.json", 2_346, "b6f19209588fcefe41f65b193fad6148446253c470d36e29441ecc5158a54e6d", "9889070ccf194bd3cab6f61954218a4058696537"),
            file("generation_config.json", 233, "02b56bd11e1cd1e363e701a85a2fd7fbaa2992ec3358c1cd7cc44ead7208f505", "6e4ef65e8cf563a9177fd933423f89eed2f74d74"),
            file("model.safetensors", 845_719_296, "3279c173daddd7186e79d652ad94022415736d3a1370625696c898429b06d6df", "dcecf2bfdf2086d661e44134663920bdaacec1e7", "3279c173daddd7186e79d652ad94022415736d3a1370625696c898429b06d6df"),
            file("tokenizer.json", 32_169_884, "c001d9ada50af662c94f5ab17ec7e09f6438d1bec8246c47fee6510693d8de35", "0645e1e88640880b270ab18c952d52b04e7e396d", "c001d9ada50af662c94f5ab17ec7e09f6438d1bec8246c47fee6510693d8de35"),
            file("tokenizer_config.json", 822, "089594a3924fcfd4cb1c596a7906fbf476193519e5198f780912eed02b177e42", "1a6bee041ca75778c514a071efbdb568b0f3d7b0"),
        ),
    )

    val totalBytes: Long get() = target.totalBytes + assistant.totalBytes

    private fun file(
        path: String,
        size: Long,
        sha256: String,
        gitOid: String,
        lfsOid: String? = null,
        source: HuggingFaceSource? = null,
    ) = LockedModelFile(path, size, sha256, gitOid, lfsOid, source)
}

object HuggingFaceCachePaths {
    fun hubRoot(): Path {
        System.getenv("HF_HUB_CACHE")?.takeIf(String::isNotBlank)?.let { return Path.of(it) }
        System.getenv("HF_HOME")?.takeIf(String::isNotBlank)?.let { return Path.of(it).resolve("hub") }
        System.getenv("XDG_CACHE_HOME")?.takeIf(String::isNotBlank)?.let {
            return Path.of(it).resolve("huggingface").resolve("hub")
        }
        return Path.of(System.getProperty("user.home"), ".cache", "huggingface", "hub")
    }

    fun repositoryRoot(repository: String): Path =
        hubRoot().resolve("models--${repository.replace("/", "--")}")

    fun snapshot(repository: String, revision: String): Path =
        repositoryRoot(repository).resolve("snapshots").resolve(revision)

    fun blob(repository: String, blobId: String): Path =
        repositoryRoot(repository).resolve("blobs").resolve(blobId)

    fun targetView(): Path = hubRoot()
        .resolve(".gem16")
        .resolve("snapshots")
        .resolve("${Gem16ModelCatalog.targetRepository.replace("/", "--")}--${Gem16ModelCatalog.targetRevision}")
}
