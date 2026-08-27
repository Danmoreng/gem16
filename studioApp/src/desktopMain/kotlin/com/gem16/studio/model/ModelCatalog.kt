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

object Gem16Qualified26BModelCatalog {
    const val targetRepository = "danmoreng/gemma-4-26B-A4B-it-GEM16"
    const val targetRevision = "b5feb4d146c5ce943160514df0c70a31059885bd"
    const val assistantRepository = "danmoreng/gemma-4-26B-A4B-it-assistant-GEM16"
    const val assistantRevision = "a741c642353ccdaefc6f987a3120f434dc9487c7"

    val target = LockedModel(
        label = "Gemma 4 26B A4B GEM16",
        repository = targetRepository,
        revision = targetRevision,
        files = listOf(
            file(".gitattributes", 95, "f15b349b90d515d0af07dc0d7da57e1a3ef9f00a723c70e8bdb8ff7fcb943b83", "c1ad41cbcce50cb9c28dbc5c41f4b17d8ce29f00"),
            file("README.md", 930, "c718e5a5e2b4a9a27eb6bc65428fe7a5b3d558378392f1cc167dbdbb0417690e", "bf0c89c228278db401faaae7ab1ca384ce424783"),
            file("chat_template.jinja", 18_683, "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4", "4741bf6e4132ba23a5537f9d6e74e9a6d613d7cd"),
            file("config.json", 4_152, "8a647d5444c9e77b03bd80ac802683ffdbf64b3d9296ca5257ced8e50349ea16", "ef0ecec886c4ee7fb9151bbd2a4395973e96c061"),
            file("gem16.lock.json", 4_231, "d7d2d30743e7c42aa55537a83f563047c9cbd2d83e0d583a2d2c8bcacbfa51a4", "d2f5aa90480db072e7b3258e16abca22e6bd93cf"),
            file("gem16_compilation.json", 3_061_406, "7e5e78b9c6f61fbe8829866395634085261e1261a8c783f69affc5a16bd1847a", "fe671accce31db603d8d139bcef8523ee3baac34"),
            file("gem16_model.json", 671, "6281508359782c2ea4977352828bcdca65987c9d74a55b553b945488ffd27a66", "7a5a33188b216561cf929af2926b648cabcc8a3d"),
            file("generation_config.json", 203, "b69207f9be617e982d13cc273cce6fd88c98dda99a4bdc5e2d52ffe0a0d9f0a9", "5a376e9fca28bfac73fe81b623bd6e4d00bb0817"),
            file("model.gem16", 14_696_668_160, "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72", "c1d972d638e8e0436e8e3ff0242cf6e9325053ea", "1ed73cf105b68db937ac0992283d31fdb2225474204341440721f41fe871bb72"),
            file("tokenizer.json", 32_169_626, "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f", "1ff9f3e3439a939b971f9919e821bf87e835a503", "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"),
            file("tokenizer_config.json", 3_729, "3ab5c7b94dc97d65ca7064496fa69b88ff875378e1cb7ee3e43070c3a8170999", "bd297dce3d66737f7d6f9136691db3ad05991a2f"),
        ),
    )

    val assistant = LockedModel(
        label = "Gemma 4 26B GEM16 fixed-D2 Assistant",
        repository = assistantRepository,
        revision = assistantRevision,
        files = listOf(
            file(".gitattributes", 101, "a0e948ca693c63e0b5139f66bd709003cbcf8bc764acb1ee687a552885c8fe2c", "0fd17d2f7c330afce8bc2b472393143fa66f68a1"),
            file("README.md", 619, "1d346a57a39bb97f74897dacb43fef636b2cc6b34a1455a43214ea775468d9d9", "74c95d6c287b95e04d549b4be9630e49b36bb664"),
            file("chat_template.jinja", 18_683, "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4", "4741bf6e4132ba23a5537f9d6e74e9a6d613d7cd"),
            file("config.json", 2_720, "d3c79600ce09c86c993c89bed7ce05baa376770967ffaa1ac9a0c29347a633e4", "049ec62ef9a48d2f0732b580d90b48cbb0f3616b"),
            file("gem16.lock.json", 1_664, "f2d2278a53ecfb3bf7a093a3207dce70148a88708a51b5050dd8d436c3d69455", "bd555c8d0653bd74ec609bb51720e462174c3545"),
            file("gem16_compilation.json", 217_371, "d2a722ccad675807ff73a1a645a0f1fbcc0fc200c402a14e823a5927618a116d", "6e938657610d03a7f30aeec5172849238d8708a0"),
            file("gem16_model.json", 613, "5769df5631d9c2d1d6cb0b4799c382b7e7aab8b340418c5e87557efab9406c39", "be525d72629e7d34bff9f67c616ddbc6b1c7df9c"),
            file("generation_config.json", 209, "fb53f4c64e58896a63472e8eb304397db4a39453e1da0f5d57625ec5a8c1050e", "76d3a00538aca6a1b85db3328e1b0f4e93316420"),
            file("model-00001-of-00001.safetensors", 258_317_280, "4d3ce2102ad0631d9e7e0586be0b108d5789cbc5b90d21b4c50613979228d927", "df6d52c496a1c194c8a167f0fbca4b52e9b2c9d1", "4d3ce2102ad0631d9e7e0586be0b108d5789cbc5b90d21b4c50613979228d927"),
            file("model.safetensors.index.json", 8_327, "8fcc4b09fb62f710558176406385e9a04423119cd6e81301171d9a64bbfd9165", "c094c640185f76249eb96a10d0d07dd7afd9ec89"),
            file("tokenizer.json", 32_169_440, "75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c", "24aa4244652e010036db5fdd29ed39b9428e6e19", "75a6583c1a418e2bbd79c60d95d28e0f5bf549ad3f2990b5bdb5238c6c2bf70c"),
            file("tokenizer_config.json", 3_023, "01f2ff1c21ef2e722891380323edcaecd9c86a776aeb9b40148e2f35e3cee4d3", "0672fbe45a4922b10e6ccd13947ecdb166bead28"),
        ),
    )

    val totalBytes: Long get() = target.totalBytes + assistant.totalBytes

    private fun file(
        path: String,
        size: Long,
        sha256: String,
        gitOid: String,
        lfsOid: String? = null,
    ) = LockedModelFile(path, size, sha256, gitOid, lfsOid)
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
