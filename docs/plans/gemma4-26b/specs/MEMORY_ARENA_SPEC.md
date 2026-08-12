# 26B memory arena specification — Fast Track R4

## Regions

Account separately:

- immutable target weights;
- target FP8 K/V;
- hidden/activation/router/permutation buffers;
- shared CUTLASS/temp workspace;
- output/sampling buffers;
- graph-private allocations;
- safety margin.

All byte arithmetic is checked and aligned. The execution plan allocates fixed addresses before prompt processing and performs no token-loop allocation.

## Base profiles

8K, 16K, 32K, 64K and a measured max candidate. One slot is supported; slot two is rejected. 32K requires 700 MiB free after warm execution; 64K+ requires 500 MiB.

## MTP extension

Add a distinct assistant arena, verifier workspace and graph-private region. Do not hide context reduction when MTP is enabled. Prefer assistant contracts without an independent long-context cache. Publish base and MTP plans separately.
