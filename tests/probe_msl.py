"""Probe which MSL primitives the CudaSift port depends on actually compile
and behave correctly under mx.fast.metal_kernel on this machine."""
import mlx.core as mx

results = {}


def probe(name, source, *, header="", inputs, output_shapes, output_dtypes,
          grid, threadgroup, expect, init_value=0):
    try:
        k = mx.fast.metal_kernel(
            name=name, input_names=[f"in{i}" for i in range(len(inputs))],
            output_names=["out"], header=header, source=source,
        )
        (out,) = k(inputs=inputs, grid=grid, threadgroup=threadgroup,
                   output_shapes=output_shapes, output_dtypes=output_dtypes,
                   init_value=init_value)
        mx.eval(out)
        ok = expect(out)
        results[name] = ("PASS" if ok else "WRONG", out.tolist()[:8])
    except Exception as e:
        results[name] = ("FAIL", str(e).strip().split("\n")[0][:160])


x = mx.arange(32, dtype=mx.float32)

# 1. simd width + shuffle_down/up/broadcast
probe(
    "simd_shuffle",
    """
    uint t = thread_position_in_threadgroup.x;
    float v = in0[t];
    float d = simd_shuffle_down(v, 1);
    float u = simd_shuffle_up(v, 1);
    float b = simd_shuffle(v, 31);
    out[t]      = (float)threads_per_simdgroup;
    out[32 + t] = d;
    out[64 + t] = (t >= 1) ? u : 0.0f;
    out[96 + t] = b;
    """,
    inputs=[x], output_shapes=[(128,)], output_dtypes=[mx.float32],
    grid=(32, 1, 1), threadgroup=(32, 1, 1),
    expect=lambda o: o[0] == 32 and o[32] == 1 and o[65] == 0 and o[96] == 31,
)

# 2. simd_any / simd_all
probe(
    "simd_any",
    """
    uint t = thread_position_in_threadgroup.x;
    bool p = in0[t] > 30.0f;          // true only for lane 31
    out[t] = simd_any(p) ? 1.0f : 0.0f;
    out[32 + t] = simd_all(p) ? 1.0f : 0.0f;
    """,
    inputs=[x], output_shapes=[(64,)], output_dtypes=[mx.float32],
    grid=(32, 1, 1), threadgroup=(32, 1, 1),
    expect=lambda o: all(v == 1 for v in o[:32].tolist()) and o[32] == 0,
)

# 3. popcount + simd_prefix_exclusive_sum  (FindPointsMultiNew compaction idiom)
probe(
    "popc_prefix",
    """
    uint t = thread_position_in_threadgroup.x;
    uint bits = (uint)in0[t];         // lane t holds value t
    uint n = popcount(bits);
    out[t] = (float)simd_prefix_exclusive_sum(n);
    """,
    inputs=[x], output_shapes=[(32,)], output_dtypes=[mx.float32],
    grid=(32, 1, 1), threadgroup=(32, 1, 1),
    expect=lambda o: o[0] == 0 and o[1] == 0 and o[2] == 1 and o[3] == 2,
)

# 4. threadgroup FLOAT atomics -- the item the doc flagged as unreliable
probe(
    "tg_atomic_float",
    """
    threadgroup atomic_float hist[4];
    uint t = thread_position_in_threadgroup.x;
    if (t < 4) atomic_store_explicit(&hist[t], 0.0f, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_fetch_add_explicit(&hist[t & 3], 1.0f, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 4) out[t] = atomic_load_explicit(&hist[t], memory_order_relaxed);
    """,
    header="#include <metal_atomic>\n",
    inputs=[x], output_shapes=[(4,)], output_dtypes=[mx.float32],
    grid=(32, 1, 1), threadgroup=(32, 1, 1),
    expect=lambda o: all(v == 8 for v in o.tolist()),
)

# 4b. CAS-loop emulation fallback for the same thing
probe(
    "tg_atomic_float_cas",
    """
    threadgroup atomic_uint hist[4];
    uint t = thread_position_in_threadgroup.x;
    if (t < 4) atomic_store_explicit(&hist[t], as_type<uint>(0.0f), memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup atomic_uint *slot = &hist[t & 3];
    uint old = atomic_load_explicit(slot, memory_order_relaxed);
    uint want;
    do { want = as_type<uint>(as_type<float>(old) + 1.0f); }
    while (!atomic_compare_exchange_weak_explicit(
        slot, &old, want, memory_order_relaxed, memory_order_relaxed));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 4) out[t] = as_type<float>(atomic_load_explicit(slot, memory_order_relaxed));
    """,
    header="#include <metal_atomic>\n",
    inputs=[x], output_shapes=[(4,)], output_dtypes=[mx.float32],
    grid=(32, 1, 1), threadgroup=(32, 1, 1),
    expect=lambda o: all(v == 8 for v in o.tolist()),
)

# 5. device-space atomic uint output (the d_PointCounter mechanism)
try:
    k = mx.fast.metal_kernel(
        name="dev_atomic_uint", input_names=["inp"], output_names=["ctr"],
        header="#include <metal_atomic>\n",
        source="""
        uint t = thread_position_in_grid.x;
        if (inp[t] >= 0.0f)
            atomic_fetch_add_explicit(&ctr[0], 1u, memory_order_relaxed);
        """,
        atomic_outputs=True,
    )
    (ctr,) = k(inputs=[x], grid=(32, 1, 1), threadgroup=(32, 1, 1),
               output_shapes=[(1,)], output_dtypes=[mx.uint32], init_value=0)
    mx.eval(ctr)
    results["dev_atomic_uint"] = ("PASS" if ctr.tolist() == [32] else "WRONG", ctr.tolist())
except Exception as e:
    results["dev_atomic_uint"] = ("FAIL", str(e).strip().split("\n")[0][:160])

# 6. fast math intrinsics used by the descriptor/orientation kernels
probe(
    "fast_math",
    """
    uint t = thread_position_in_grid.x;
    float v = in0[t] + 1.0f;
    out[t]      = metal::fast::exp(-v);
    out[32 + t] = metal::rsqrt(v);
    out[64 + t] = metal::fast::divide(1.0f, v);
    out[96 + t] = metal::precise::atan2(v, 1.0f);
    """,
    inputs=[x], output_shapes=[(128,)], output_dtypes=[mx.float32],
    grid=(32, 1, 1), threadgroup=(32, 1, 1),
    expect=lambda o: abs(o[32].item() - 1.0) < 1e-5,
)

width = max(len(n) for n in results)
for name, (status, detail) in results.items():
    print(f"{name:<{width}}  {status:<5}  {detail}")
