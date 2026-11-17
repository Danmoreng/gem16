"""Offline-only experimental SM120 D3 global attention; never imported at runtime."""
import triton
import triton.language as tl


@triton.jit
def gem16_d3_global_fp8(Q, K, V, KS, VS, CONTROL, PARTIAL, LSE, CAPACITY,
                        HEAD_DIM: tl.constexpr = 512,
                        HEADS: tl.constexpr = 16,
                        SEGMENTS: tl.constexpr = 16,
                        TILE: tl.constexpr = 32,
                        USE_TMA: tl.constexpr = False):
    segment = tl.program_id(0)
    row = tl.program_id(1)
    # DecodeControl.position is uint64 at byte offset8. Four causal queries
    # start at controls[0].position, just as in the exact native verifier.
    tokens = tl.load(CONTROL + 1) + row + 1
    head = tl.arange(0, HEADS)
    dim = tl.arange(0, HEAD_DIM)
    key = tl.arange(0, TILE)
    q = tl.load(Q + row * HEADS * HEAD_DIM + head[:, None] * HEAD_DIM + dim[None, :])
    q = tl.minimum(tl.maximum(q, -448.0), 448.0).to(tl.float8e4nv)
    kscale = tl.load(KS).to(tl.float32)
    vscale = tl.load(VS).to(tl.float32)
    tiles = tl.cdiv(tokens, SEGMENTS * TILE)
    first = segment * tiles
    maximum = tl.full((HEADS,), float('-inf'), tl.float32)
    denominator = tl.zeros((HEADS,), tl.float32)
    accumulator = tl.zeros((HEADS, HEAD_DIM), tl.float32)
    if USE_TMA:
        # Bound partial TMA tiles by the physical allocation. The causal
        # mask below is per query and may end earlier within the same tile.
        kd = tl.make_tensor_descriptor(K, shape=[CAPACITY, HEAD_DIM],
                                      strides=[HEAD_DIM, 1],
                                      block_shape=[TILE, HEAD_DIM])
        vd = tl.make_tensor_descriptor(V, shape=[CAPACITY, HEAD_DIM],
                                      strides=[HEAD_DIM, 1],
                                      block_shape=[TILE, HEAD_DIM])
    for tile in range(first, first + tiles):
        position = tile * TILE + key
        valid = position < tokens
        if USE_TMA:
            offset = (tile * TILE).to(tl.int32)
            k = kd.load([offset, 0]).T
            v = vd.load([offset, 0])
        else:
            k = tl.load(K + dim[:, None] + position[None, :] * HEAD_DIM,
                        mask=valid[None, :], other=0.0)
            v = tl.load(V + position[:, None] * HEAD_DIM + dim[None, :],
                        mask=valid[:, None], other=0.0)
        scores = tl.dot(q, k) * kscale
        scores = tl.where(valid[None, :], scores, float('-inf'))
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        # An entirely empty segment/tile must not create (-inf)-(-inf) NaNs.
        safe_maximum = tl.where(new_maximum == float('-inf'), 0.0, new_maximum)
        correction = tl.exp(maximum - safe_maximum)
        probability = tl.exp(scores - safe_maximum[:, None])
        denominator = denominator * correction + tl.sum(probability, 1)
        accumulator = accumulator * correction[:, None]
        accumulator += tl.dot((probability * 256.0).to(tl.float8e4nv), v) * (1.0 / 256.0)
        maximum = new_maximum
    inverse = tl.where(denominator > 0.0, 1.0 / denominator, 0.0)
    output = accumulator * inverse[:, None] * vscale
    # Match native fixed-row partial layout [row,split,head,dimension].
    offsets = ((row * SEGMENTS + segment) * HEADS + head[:, None]) * HEAD_DIM + dim[None, :]
    tl.store(PARTIAL + offsets, output)
    lse = tl.where(denominator > 0.0, maximum + tl.log(denominator), float('-inf'))
    tl.store(LSE + (row * SEGMENTS + segment) * HEADS + head, lse)
