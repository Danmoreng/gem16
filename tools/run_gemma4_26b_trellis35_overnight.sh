#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build/Linux/blackwell-release"
source_root="${repo_root}/models/checkpoints/google-gemma-4-26b-a4b-it-qat-bf16-f1e06dc"
base_model="${repo_root}/artifacts/raw/m25/hf-qualified-checkpoints-final-test-2026-08-27/target"
output_dir="${repo_root}/models/checkpoints/google-gemma-4-26b-a4b-it-trellis35-w4a8-wp2"
first_layer="${GEM16_TRELLIS35_FIRST_LAYER:-0}"
last_layer="${GEM16_TRELLIS35_LAST_LAYER:-29}"

cd "${repo_root}"

cmake --preset blackwell-release
cmake --build "${build_dir}" --target \
  gem16-trellis35-calibration \
  gem16-trellis35-expand-down \
  gem16-trellis35-quantize \
  gem16-trellis35-ldlq \
  -j2

mkdir -p "${output_dir}"

python3 tools/generate_gemma4_26b_trellis35_checkpoint.py \
  --source-root "${source_root}" \
  --reference-model "${base_model}" \
  --output-dir "${output_dir}" \
  --native-calibration "${build_dir}/bin/gem16-trellis35-calibration" \
  --native-expand-down "${build_dir}/bin/gem16-trellis35-expand-down" \
  --native-quantize "${build_dir}/bin/gem16-trellis35-quantize" \
  --native-ldlq "${build_dir}/bin/gem16-trellis35-ldlq" \
  --first-layer "${first_layer}" \
  --last-layer "${last_layer}"

if [[ "${first_layer}" == "0" && "${last_layer}" == "29" ]]; then
  python3 tools/package_gemma4_26b_trellis35_checkpoint.py \
    --base-model "${base_model}" \
    --checkpoint "${output_dir}"
  python3 tools/verify_gemma4_26b_trellis35_checkpoint.py \
    --checkpoint "${output_dir}" \
    --report "${output_dir}/checkpoint-verification.json"
  sha256sum \
    "${output_dir}/trellis35-checkpoint.json" \
    "${output_dir}/non-routed.gem16" \
    "${output_dir}/checkpoint-verification.json" \
    > "${output_dir}/FINAL_SHA256SUMS"
fi

echo "trellis35_overnight_ok output=${output_dir} layers=${first_layer}-${last_layer}"
