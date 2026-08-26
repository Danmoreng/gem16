# Task-quality benchmarks

`tools/benchmark_quality.py` measures task accuracy through an OpenAI-compatible
endpoint. It does not require token-for-token agreement between runtimes.
Prompts, answer extraction, and scoring come from `sgl-eval` at commit
`fe0df4731526a73e915bedc1e2750ecb79274996`, whose NeMo-Skills scoring snapshot
is vendored and recorded in every result.

The first supported matrix is deliberately small:

| Task | Examples | Default reasoning | External anchor |
|---|---:|---|---:|
| GSM8K | 1,319 | none | AxionML 12B NVFP4 96.12% (different checkpoint) |
| GPQA Diamond | 198 | high | Google Gemma 4 12B 78.8% |
| AIME 2026 | 30 | high | Google Gemma 4 12B 77.5% |

## Paired Gemma 4 26B smoke

The repository provides a one-command 20-question GSM8K smoke for the frozen
Gemma 4 26B candidate and Google's pinned official QAT Q4_0 GGUF reference:

```bash
./scripts/run-gemma4-26b-gsm8k-smoke.sh
```

It creates the pinned quality environment when necessary, starts the pinned
llama.cpp reference once, runs the first 20 GSM8K questions, releases that
server, then starts gem16 and runs the identical questions. The final paired
report includes both accuracies, outcome counts, changed IDs, and the exact
McNemar result. Set `OUTPUT_DIR` to choose a different result directory; all
server, model, Python, port, and timeout paths have explicit environment
overrides documented by `--help`.

This smoke is diagnostic. M19 requires complete task reports with the frozen
artifact and reference identities.

Run the complete paired 1,319-question GSM8K comparison with an explicit,
stable result directory:

```bash
OUTPUT_DIR="$PWD/benchmarks/results/<date>/<revision>/local-gsm8k-full1319" \
  ./scripts/run-gemma4-26b-gsm8k-full.sh
```

The reference and gem16 servers are loaded sequentially and each remains
resident for its complete pass. Both passes journal every completed question;
rerunning the same command with the same `OUTPUT_DIR` resumes the unfinished
pass and skips any already complete pass before regenerating the paired
comparison. Keep that explicit path unchanged when continuing on a later day;
the date-scoped default would otherwise select a new directory after midnight.

Google's exact internal protocols are not fully published, so those model-card
numbers are plausibility anchors rather than exact pass/fail thresholds. The
primary engine-quality comparison is paired task accuracy between the exact
pinned Unsloth checkpoint in vLLM and in gem16.

## Environment

`sgl-eval` supports Python 3.10 through 3.13. Use a separate environment rather
than the repository's Python 3.14 tooling interpreter:

```bash
python3.13 -m venv .venv-quality
.venv-quality/bin/python -m pip install -r tools/requirements-quality.txt
```

GPQA is gated. Accept its Hugging Face terms and set `HF_TOKEN` before the first
run. GSM8K is fetched from the exact OpenAI repository commit and verified by
SHA-256. AIME data and graders are bundled in the pinned `sgl-eval` revision.

## Reasoning and generation profiles

Use two distinct comparisons; do not mix their results.

### Product-quality gate (default)

The default `--generation checkpoint` uses the checkpoint-recommended
`temperature=1.0`, `top_p=0.95`, and `top_k=64`. GSM8K runs without thinking
to retain a link to the published 12B NVFP4 result. GPQA and AIME use `high`,
gem16's bounded 8,192-token reasoning profile and the mode used by Google's
published model results. This is the primary product-quality run. GPQA and
AIME reserve up to 16,384 output positions: an 8,192-token high-reasoning cap
alone is insufficient because the model still needs room to emit its scored
answer after a forced reasoning close.

Start gem16 without `--greedy`; MTP D2 may remain enabled because exact target
verification preserves the ordinary sampled decision for the configured seed:

```bash
build/Linux/blackwell-release/bin/gem16-server \
  --model models/checkpoints/unsloth-gemma-4-12b-it-NVFP4-b1f6497 \
  --assistant-model models/checkpoints/google-gemma-4-12B-it-assistant-364bd03 \
  --mtp-draft-tokens 2 --max-context 32768 --max-sessions 1
```

Smoke-test twenty questions before a complete run:

```bash
.venv-quality/bin/python tools/benchmark_quality.py \
  --benchmark gsm8k --backend gem16 \
  --base-url http://127.0.0.1:8080/v1 \
  --num-examples 20 \
  --output benchmarks/results/<date>/<sha>/<machine>/gem16-gsm8k-smoke.json
```

Full run:

```bash
.venv-quality/bin/python tools/benchmark_quality.py \
  --benchmark gsm8k --backend gem16 \
  --base-url http://127.0.0.1:8080/v1 \
  --output benchmarks/results/<date>/<sha>/<machine>/gem16-gsm8k.json
```

For a vLLM endpoint serving the exact same Unsloth checkpoint, change
`--backend openai` and the URL. The generic adapter sends greedy sampling and
`chat_template_kwargs.enable_thinking`; gem16 instead receives its supported
`reasoning_effort` field. Dataset, prompt, scoring, seed, output cap, and task
order remain equal.

Compare complete paired runs:

```bash
.venv-quality/bin/python tools/compare_quality_benchmarks.py \
  --reference benchmarks/results/.../vllm-gsm8k.json \
  --candidate benchmarks/results/.../gem16-gsm8k.json \
  --output benchmarks/results/.../gsm8k-comparison.json
```

The comparison reports both accuracies, absolute delta, relative retention,
the four paired outcome counts, changed question IDs, and the exact two-sided
McNemar p-value.

### Deterministic engine-isolation run

Pass `--generation greedy` and start both runtimes in greedy mode when a second,
variance-free engine comparison is useful. This isolates arithmetic/runtime
differences but is not the primary model-quality result.

The current gem16 server fixes its seed at session creation, so repeated sampled
runs from one process are deterministic and must not be presented as independent
Google-style samples. Use one sampled pass per task until independent
per-request seeds are implemented.

## Runtime estimates

The harness prints an estimate before every run. Override its assumptions with
`--average-output-tokens`, `--tokens-per-second`, and
`--per-request-overhead-seconds`, or use `--estimate-only`.

On the current roughly 35 tok/s ordinary path, approximate serial times are:

| Run | Assumption | Approximate wall time |
|---|---|---:|
| GSM8K full | 128-256 output tokens/question | 1.4-2.9 h plus slot/prompt overhead |
| GPQA, one sampled pass | about 8K output tokens/question | about 6.5-8 h plus overhead |
| AIME, one sampled pass | up to 8K output tokens/question | about 2 h plus overhead |
| GPQA, 8 sampled repeats | same range | about 52-64 h |
| AIME, 16 sampled repeats | same range | about 32 h |

MTP can reduce these times when acceptance is representative, but quality
results count only target-verified answers. Start with 20-example smoke runs,
then GSM8K, then one-pass GPQA. A full repeated AIME/GPQA characterization is an
overnight or multi-day job on one batch-one GPU.

Every output includes all questions, raw responses, extracted answers, binary
scores, token usage, finish reason, timing, endpoint health, dataset revision,
and benchmark implementation provenance. Existing files and incremental sample
directories are never overwritten.

## Resuming long runs

Add `--resume` to the first and every later invocation of a long run. Resume is
currently restricted to the M19 protocol of one repeat and one request thread:

```bash
.venv-quality/bin/python tools/benchmark_quality.py \
  --benchmark gsm8k --backend gem16 \
  --base-url http://127.0.0.1:8080/v1 \
  --repeats 1 --threads 1 --resume \
  --output benchmarks/results/<date>/<sha>/<machine>/gem16-gsm8k.json
```

Each scored sample is appended and synchronized immediately in the adjacent
`<output-stem>.samples/output-rs0.jsonl` journal. A `resume-state.json` freezes
the benchmark, model ID, dataset, evaluator revision, generation protocol, and
planned question IDs before the first request. Reusing the same command and
output path skips completed IDs. Any change to that contract fails visibly and
requires a new output path. The top-level JSON is derived from the journal and
may be safely regenerated after an interrupted attempt.
