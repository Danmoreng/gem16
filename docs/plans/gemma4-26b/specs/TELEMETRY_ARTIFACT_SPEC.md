# Telemetry and evidence artifact specification

## Goal

Every memory, quality and performance claim must be traceable to machine-readable raw evidence.

## Common envelope

All JSON reports begin with:

```json
{
  "schema_version": 1,
  "created_utc": "...",
  "code": {
    "repository": "Danmoreng/gem16",
    "commit": "...",
    "dirty": false
  },
  "model": {
    "profile": "...",
    "artifact_lock_sha256": "...",
    "source_lock_sha256": "..."
  },
  "toolchain_lock_sha256": "...",
  "machine_id": "...",
  "command": ["..."],
  "environment": {}
}
```

## Environment

Record:

- OS/kernel;
- CPU and RAM;
- swap;
- GPU name/UUID/compute capability;
- total/free VRAM;
- driver/runtime/toolkit;
- power limit/profile;
- clocks;
- VBIOS when available;
- desktop/display state if relevant;
- relevant environment variables.

## Runtime path

Record:

- model variant;
- head format;
- attention path;
- MoE decode/prefill path;
- KV mode;
- prefill chunk;
- CUDA Graph status;
- native instruction capability;
- fallback count;
- token-loop allocation flag.

## Memory

Named allocator accounting:

```json
{
  "immutable": {},
  "kv": {},
  "prefill_workspace": {},
  "decode_workspace": {},
  "graph_private": {},
  "total_named": 0,
  "cuda_mem_get_info_delta": 0,
  "sampled_process_peak": 0,
  "margin": 0
}
```

Keep decimal bytes and add MiB/GiB only in human summaries.

## Performance raw run

```json
{
  "run": 1,
  "warmup": false,
  "prompt_tokens": 8192,
  "output_forwards": 256,
  "prompt_ms": 0.0,
  "ttft_ms": 0.0,
  "decode_ms": 0.0,
  "decode_tps": 0.0,
  "itl_ms": [],
  "output_checksum": 0,
  "fallback_count": 0
}
```

Summary calculations can be regenerated from raw runs.

## Quality

Store per-example and aggregate data. Include exact token IDs or their content-addressed manifest identity.

Do not store only percentages.

## GPU telemetry stream

CSV or JSONL fields:

```text
timestamp
process_vram_mib
gpu_util
memory_util
power_w
sm_clock_mhz
memory_clock_mhz
temperature_c
throttle_reasons
```

Synchronize clocks with benchmark events or include markers.

## Nsight and disassembly

Each trace gets a sidecar:

- command;
- binary hash;
- model hash;
- capture range;
- tool version;
- kernel filter;
- interpretation notes.

Do not commit enormous binary traces to Git unless policy allows; retain content-addressed external location plus checksum.

## Integrity

Every evidence directory includes:

```text
commands.txt
environment.json
SHA256SUMS
```

Raw evidence is immutable after publication. Corrections create a new directory and explanatory note.

## Human-readable summary

Human documents quote values from machine-readable reports. Prefer a generation script to prevent transcription errors.

## Privacy/security

Do not include:

- Hugging Face tokens;
- personal home paths where avoidable;
- API keys;
- private prompt content without consent;
- full environment dumps containing secrets.

Redact before hashing/publishing, not afterward.

## Actual-path and lifecycle amendments

Runtime evidence adds:

```json
{
  "resolved_dispatch": {
    "attention_prefill": "...",
    "attention_decode": "...",
    "moe_decode": "...",
    "moe_prefill": "...",
    "embedding_head": "..."
  },
  "cuda_graph": {
    "enabled": true,
    "first_demotion_reason": "none"
  },
  "lifecycle": {
    "engine_cycle": 1,
    "sticky_cuda_error_after_teardown": "cudaSuccess",
    "device_bytes_recovered": 0
  }
}
```

Set the dispatch value at the branch that actually wins. Do not reconstruct it later from configuration.
