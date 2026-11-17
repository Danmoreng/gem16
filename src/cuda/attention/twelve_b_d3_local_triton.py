"""Offline-only experimental T4 local FP8 attention; native1024 ring plus drafts."""
import triton
import triton.language as tl


@triton.jit
def gem16_d3_local_fp8(Q, K, V, SK, SV, KS, VS, CONTROL, PARTIAL, LSE,
                      SEGMENTS: tl.constexpr = 4, TILE: tl.constexpr = 64):
    kvhead = tl.program_id(0)
    segment = tl.program_id(1)
    start = tl.load(CONTROL + 1).to(tl.int32)
    first = tl.maximum(start - 1023, 0)  # CONTROL is signedint64; avoids unsigned underflow.
    total = start + 4 - first
    m = tl.arange(0, 16)
    row, head = m // 2, kvhead * 2 + m % 2
    dim, column = tl.arange(0, 256), tl.arange(0, TILE)
    q = tl.load(Q + (row[:, None] * 16 + head[:, None]) * 256 + dim[None, :],
                mask=row[:, None] < 4, other=0.0)
    q = tl.minimum(tl.maximum(q, -448.0), 448.0).to(tl.float8e4nv)
    kscale, vscale = tl.load(KS).to(tl.float32), tl.load(VS).to(tl.float32)
    maximum = tl.full((16,), float('-inf'), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, 256), tl.float32)
    for tile in range(segment, tl.cdiv(total, TILE), SEGMENTS):
        absolute = first + tile * TILE + column
        history = absolute < start
        valid = absolute < start + 4
        cache_offset = ((absolute % 1024) * 8 + kvhead) * 256
        draft_offset = ((absolute - start) * 8 + kvhead) * 256
        key_base = tl.where(history, K + cache_offset, SK + draft_offset)
        value_base = tl.where(history, V + cache_offset, SV + draft_offset)
        k = tl.load(key_base[:, None] + dim[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(value_base[:, None] + dim[None, :], mask=valid[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k)) * kscale
        allowed = (row[:, None] < 4) & valid[None, :] & \
            (absolute[None, :] <= start + row[:, None]) & \
            (absolute[None, :] >= tl.maximum(start + row[:, None] - 1023, 0))
        scores = tl.where(allowed, scores, float('-inf'))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        safe = tl.where(next_maximum == float('-inf'), 0.0, next_maximum)
        correction = tl.exp(maximum - safe)
        probability = tl.exp(scores - safe[:, None])
        denominator = denominator * correction + tl.sum(probability, 1)
        accumulator = accumulator * correction[:, None] + \
            tl.dot((probability * 256.0).to(tl.float8e4nv), v) * (1.0 / 256.0)
        maximum = next_maximum
    inverse = tl.where(denominator > 0.0, 1.0 / denominator, 0.0)
    offsets = ((row[:, None] * SEGMENTS + segment) * 16 + head[:, None]) * 256 + dim[None, :]
    tl.store(PARTIAL + offsets, accumulator * inverse[:, None] * vscale,
             mask=row[:, None] < 4)
    tl.store(LSE + (row * SEGMENTS + segment) * 16 + head,
             tl.where(denominator > 0.0, maximum + tl.log(denominator), float('-inf')),
             mask=row < 4)
