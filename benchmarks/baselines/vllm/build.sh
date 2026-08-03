#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../.." && pwd)"
python_bin="${VLLM_BASE_PYTHON:-}"
if [[ -z "${python_bin}" ]]; then
  if command -v python3.13 >/dev/null 2>&1; then
    python_bin="$(command -v python3.13)"
  elif [[ -x "${HOME}/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/bin/python3.13" ]]; then
    python_bin="${HOME}/.local/share/uv/python/cpython-3.13.14-linux-x86_64-gnu/bin/python3.13"
  else
    echo "error: CPython 3.13.14 not found; set VLLM_BASE_PYTHON" >&2
    exit 2
  fi
fi
"${python_bin}" - <<'PY'
import sys
if sys.version_info[:3] != (3, 13, 14):
    raise SystemExit(f"expected CPython 3.13.14, got {sys.version.split()[0]}")
PY

env_dir="${VLLM_ENV_DIR:-${repo_root}/third_party/cache/vllm-0.26.0-env}"
patch_file="${script_dir}/patches/gemma4-mtp-suppress-graph.patch"
original_sha256="4eee061c81430be28f029ed66360887a57f8711a75c863067d30e3840a488918"
patched_sha256="2436a940cc7f525880588392a08f5f2b509b51f91394d6666dba181302cf92f7"

if [[ ! -x "${env_dir}/bin/python" ]]; then
  "${python_bin}" -m venv "${env_dir}"
fi

"${env_dir}/bin/python" -m pip install --upgrade pip wheel 'setuptools==80.10.2'
"${env_dir}/bin/python" -m pip install \
  'vllm==0.26.0' \
  'torch==2.11.0' \
  'transformers==5.14.1' \
  'compressed-tensors==0.17.0'

site_dir="$("${env_dir}/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
source_file="${site_dir}/vllm/model_executor/models/gemma4_mtp.py"
actual_sha256="$(sha256sum "${source_file}" | cut -d' ' -f1)"
if [[ "${actual_sha256}" == "${original_sha256}" ]]; then
  patch -p1 -d "${site_dir}" < "${patch_file}"
elif [[ "${actual_sha256}" != "${patched_sha256}" ]]; then
  echo "error: unexpected vLLM Gemma 4 MTP source checksum: ${actual_sha256}" >&2
  exit 2
fi

"${env_dir}/bin/python" - <<'PY'
import importlib.metadata
from pathlib import Path

import torch

expected = {
    "vllm": "0.26.0",
    "torch": "2.11.0",
    "transformers": "5.14.1",
    "compressed-tensors": "0.17.0",
    "setuptools": "80.10.2",
}
actual = {name: importlib.metadata.version(name) for name in expected}
if actual != expected:
    raise SystemExit(f"version mismatch: expected {expected}, got {actual}")
source = Path(importlib.metadata.distribution("vllm").locate_file(
    "vllm/model_executor/models/gemma4_mtp.py"
))
if "for token_id in self._suppress_token_ids:" not in source.read_text(encoding="utf-8"):
    raise SystemExit("vLLM Gemma 4 graph-safe suppression patch is not applied")
if not torch.cuda.is_available():
    raise SystemExit("vLLM Torch cannot access CUDA")
print(actual)
print(f"torch_cuda={torch.version.cuda}")
print(f"environment={Path(__import__('sys').prefix)}")
PY
