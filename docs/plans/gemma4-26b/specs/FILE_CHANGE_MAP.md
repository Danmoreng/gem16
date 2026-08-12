# Repository change map — Fast Track R4

This map is directional; inspect the actual tree before editing.

| Lane | Primary areas | Milestones |
|---|---|---|
| Compiler | `src/compiler/**`, compiler CLI/tests, `tools/gem16_compile/**` | M06–M08 |
| MoE oracle | new numeric oracle and tests | M10 |
| CUDA MoE | `src/cuda/moe/**` | M11, M14, M15 |
| Attention/KV | model traits, `src/cuda/attention/**`, `kv_cache/**`, `rope/**` | M12 |
| Memory | `src/runtime/memory_plan*`, internal counters/tests | M09, M21, M25 |
| Head | compiler head binding, embedding/output kernels | M07, M16, M25 |
| Integration | target model, inference engine/orchestration | M08, M13, M17, M25 |
| Harnesses | validation/evaluation/benchmark tools and prompts | M13, M19–M21 |
| Product | CLI/server; Studio optional | M22, M25 |

Shared public structs and integration orchestration are owned by the lead agent. Parallel sub-agents receive exact writable paths before work begins.
