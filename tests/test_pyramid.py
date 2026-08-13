"""Validate the MLX scale-space against an independent NumPy implementation
that reproduces CudaSift's index arithmetic literally (clamped gather, no
convolution primitive), so a mistake in the conv2d/padding formulation shows up.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import numpy as np

from metalsift import pyramid as P


def ref_separable(img, taps, stride=1):
    """Clamped separable filter written as explicit gathers, the way the CUDA
    kernels do it (max(min(i, n-1), 0))."""
    h, w = img.shape
    r = (len(taps) - 1) // 2
    oh, ow = (h + stride - 1) // stride, (w + stride - 1) // stride
    tmp = np.zeros((h, ow), np.float64)
    for u in range(ow):
        for j, t in enumerate(taps):
            xi = np.clip(u * stride + j - r, 0, w - 1)
            tmp[:, u] += t * img[:, xi]
    out = np.zeros((oh, ow), np.float64)
    for v in range(oh):
        for j, t in enumerate(taps):
            yi = np.clip(v * stride + j - r, 0, h - 1)
            out[v] += t * tmp[yi]
    return out


def rel_err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-12)


def main():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (97, 131)).astype(np.float32)   # odd dims on purpose
    mimg = mx.array(img)
    fails = []

    def check(name, got, want, tol=2e-6):
        e = rel_err(got, want)
        status = "PASS" if e < tol else "FAIL"
        if status == "FAIL":
            fails.append(name)
        print(f"  {name:<34} {status}  rel_err={e:.3e}")

    print("scale-space vs. NumPy reference")

    # LowPass, initBlur=1.0 as in mainSift.cpp
    taps = P.lowpass_taps(1.0)
    check("lowpass(sigma=1.0)", P.lowpass(mimg, 1.0), ref_separable(img, taps))

    # ScaleDown
    st_np = P.scaledown_taps(0.5)
    ref = ref_separable(img, st_np, stride=2)[: 97 // 2, : 131 // 2]
    check("scale_down", P.scale_down(mimg, st_np), ref)

    # LaplaceMulti: all 8 blurs and the 7 differences.
    # The DoG is a difference of two ~200-magnitude blurs, so it loses ~2
    # decimal digits to cancellation in float32. Scoring the error against the
    # DoG's own range would just measure that cancellation; the meaningful
    # bound is a few float32 ulps of the operands being differenced.
    kern = P.laplace_taps(5)[5]
    dog = np.asarray(P.laplace_multi_ops(mimg, kern))
    blurs = np.stack([ref_separable(img, kern[i]) for i in range(P.LAPLACE_S)])
    ref_dog = blurs[1:] - blurs[:-1]
    ulps = np.abs(dog - ref_dog).max() / (np.finfo(np.float32).eps * np.abs(blurs).max())
    ok = ulps < 8
    print(f"  {'laplace_multi (7 DoG planes)':<34} {'PASS' if ok else 'FAIL'}  "
          f"err={ulps:.1f} ulp of blur magnitude")
    if not ok:
        fails.append("laplace_multi")

    # The fused Metal kernels must agree with the ops formulation. Odd
    # dimensions are the interesting case: they exercise the tail guards in
    # both the halo load and the output write.
    from metalsift import msl
    lt, st = mx.array(taps), mx.array(P.scaledown_taps(0.5))
    for name, got, want in (
        ("msl.lowpass", msl.lowpass(mimg, lt), P.lowpass(mimg, 1.0)),
        ("msl.scale_down", msl.scale_down(mimg, st), P.scale_down(mimg, st_np)),
    ):
        got, want = np.asarray(got), np.asarray(want)
        ok = got.shape == want.shape and rel_err(got, want) < 2e-6
        print(f"  {name + ' vs ops':<34} {'PASS' if ok else 'FAIL'}  "
              f"rel_err={rel_err(got, want):.3e} shape={got.shape}")
        if not ok:
            fails.append(name)

    msl_dog = np.asarray(msl.laplace_multi(mimg, mx.array(kern), 131, 97))
    u2 = np.abs(msl_dog - dog).max() / (np.finfo(np.float32).eps * np.abs(blurs).max())
    ok2 = u2 < 8
    print(f"  {'msl.laplace_multi vs ops':<34} {'PASS' if ok2 else 'FAIL'}  "
          f"err={u2:.1f} ulp of blur magnitude")
    if not ok2:
        fails.append("msl.laplace_multi")

    # Impulse response: confirms ScaleDown samples centred on (2u, 2v) with no
    # half-pixel shift -- the thing that silently drifts keypoints if wrong.
    imp = np.zeros((32, 32), np.float32)
    imp[10, 14] = 1.0
    d = np.asarray(P.scale_down(mx.array(imp), st_np))
    peak = np.unravel_index(np.argmax(d), d.shape)
    ok = peak == (5, 7)
    print(f"  {'scale_down impulse centring':<34} {'PASS' if ok else 'FAIL'}  "
          f"peak at {peak}, expected (5, 7)")
    if not ok:
        fails.append("scale_down impulse centring")

    # Filter taps must be normalised, or DoG magnitudes drift off `thresh`
    for o in (1, 3, 5):
        k = P.laplace_taps(5)[o]
        s = np.abs(k.sum(axis=1) - 1.0).max()
        print(f"  {'laplace taps sum to 1 (octave %d)' % o:<34} "
              f"{'PASS' if s < 1e-6 else 'FAIL'}  max_dev={s:.2e}")
        if s >= 1e-6:
            fails.append(f"laplace taps octave {o}")

    print("FAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
