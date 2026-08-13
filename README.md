# metalSIFT

A port of [CudaSift](https://github.com/Celebrandil/CudaSift) (Pascal branch, MIT)
to Apple Silicon via MLX + custom Metal kernels. Built and measured on an M5.

This is a derivative work of CudaSift, copyright (c) 2017 Mårten Björkman, and
is distributed under the same MIT licence with that copyright notice retained.
The algorithm, kernel structure and parameter choices are his; what is new here
is the Metal/MLX implementation.

## Setup

Upstream CudaSift is not vendored — its `data/` images are what everything runs
against, and `cudaSiftD.cu` is what the port is line-referenced to:

```
./scripts/fetch_cudasift.sh
mise install                       # Python 3.13 + .venv
mise exec -- uv pip install mlx numpy opencv-python-headless
```

```
mise exec -- python demo.py                        # mainSift.cpp equivalent
mise exec -- python tests/test_cudasift_suite.py   # CudaSift's own tests, ported  -> 29/29
mise exec -- python tests/test_pyramid.py          # scale-space vs. NumPy reference
mise exec -- python tests/test_endtoend.py         # synthetic warp, OpenCV agreement
mise exec -- python tests/probe_msl.py             # which MSL primitives compile here
```

## Results (1920x1080, 5 octaves, initBlur 1.0, thresh 3.0)

| stage | ms |
|---|---|
| **full extraction** | **5.44** |
| matching, 2061 x 2240 descriptors | 1.47 |

Per stage, summed over all five octaves (each measured with its own sync, so
these sum to more than the pipeline total — lazy evaluation overlaps them):

| stage | ms | where |
|---|---|---|
| LowPass + 5x ScaleDown | 2.36 | MLX ops |
| LaplaceMulti (DoG) | 1.91 | Metal |
| FindPointsMulti | 1.38 | Metal |
| ComputeOrientations | 0.68 | Metal |
| duplicate compaction | 0.86 | Metal |
| ExtractSiftDescriptors | 0.90 | Metal |

For reference, CudaSift reports 1.7 ms at this resolution on a GTX 1080 Ti.

## Layout

```
metalsift/pyramid.py   scale-space as MLX ops (LowPass, ScaleDown, ScaleUp)
metalsift/msl.py       Metal kernels: LaplaceMulti, FindPoints, Orientations,
                       compaction, Descriptors
metalsift/sift.py      the ExtractSift / ExtractSiftLoop / ExtractSiftOctave pipeline
metalsift/matching.py  matching and RANSAC homography, pure MLX ops
```

## The three CUDA features with no MLX equivalent

**Textures.** Every subpixel read in the orientation and descriptor kernels is
`tex2D<float>` with `cudaFilterModeLinear` + `cudaAddressModeClamp`. MLX hands
you `device float*`, so `texfetch` in `msl.py` does the 4-tap bilinear fetch by
hand. The half-pixel convention matters: CUDA's linear fetch at `(x, y)` samples
around `(x-0.5, y-0.5)`, which is why `ExtractSiftDescriptorsCONSTNew` adds
`+0.5` to its coordinates and `ComputeOrientationsCONST` does not. That
asymmetry is real — CudaSift's orientation window genuinely sits half a pixel
off from its descriptor window — and is reproduced rather than "fixed", since
correcting it would change which keypoints come out.

**Threadgroup float atomics.** Metal's `atomic_float` overloads are
`device`-address-space only; `threadgroup atomic_float` is a *compile error*,
not merely unreliable (`tests/probe_msl.py` demonstrates it). Both
`ExtractSiftDescriptorsCONSTNew` (128-bin trilinear scatter) and
`ComputeOrientationsCONST` (32-bin gradient histogram) need them, so
`tg_add_float` emulates them with a CAS loop on `threadgroup atomic_uint`.

**Mutable device globals.** CudaSift's multi-octave design hinges on
`__device__ unsigned int d_PointCounter[17]` being mutated across kernel
launches, each octave reading where the previous one stopped appending. Rather
than thread that functionally, each octave here gets its own keypoint buffer
and counter, concatenated at the end. That costs exactly one device sync for
the whole extraction.

A related hazard the counter chain hides: `ComputeOrientationsCONST` appends
*new* keypoints for secondary orientation peaks into the same buffer it is
reading from, while other threadgroups are still reading it. MLX inputs are
const and outputs are distinct buffers, so this is split — the orientation
kernel emits a second orientation per slot, and a separate compaction pass
copies `[0, n)` and appends duplicates at `>= n`. The two regions are disjoint
by construction, so one kernel does both without aliasing.

## Deviations from CudaSift

- **Border keypoints are rejected.** `FindPointsMultiNew` reads one pixel
  outside the image when refining a candidate on the border; in CUDA that lands
  in pitch padding harmlessly, but MLX arrays are tight. Candidates within 1 px
  of the edge are dropped instead.
- **Overflow drops rather than overwrites.** CudaSift clamps an overflowing
  keypoint index to `maxPts-1`, repeatedly overwriting the last slot. This skips.
- **Matching covers all descriptors.** `FindMaxCorr10` loops
  `bp2 < numPts2 - 32 + 1`, silently ignoring up to 31 trailing descriptors of
  the second set. The matmul formulation does not, so match counts differ
  slightly from the original by construction.
- **RANSAC samples distinct positions.** Orientation doubling puts several
  keypoints at one location; drawing two of them gives duplicate rows and an
  exactly singular 8x8 system.
- `mainSift.cpp`'s `imgSet` 1 path and the pitch alignment (`iAlignUp(w, 128)`)
  are gone — MLX arrays are row-contiguous.

## Two things worth knowing about MLX on this hardware

**`mx.conv2d` is the wrong tool for separable filters.** At 1920x1080 a
9-tap depthwise `conv2d` over 8 channels takes **18.7 ms**, and even a
single-channel 9-tap pass takes 2.4 ms. The same filter written as fused
shifted multiply-adds under `mx.compile` takes 1.26 ms, and the custom Metal
port of `LaplaceMultiMem` takes **1.11 ms**. Building the DoG pyramid from
`conv2d` — the obvious pure-MLX approach — cost 38 of the original 45 ms.
`LowPass`/`ScaleDown` still use the compiled shift form; only the 8-scale DoG
justified a custom kernel.

**`mx.matmul` is not full float32 on the GPU.** It dispatches to a
reduced-precision hardware path: ~8e-4 relative error, versus 4e-7 for the same
matmul on `mx.cpu` and 1e-7 for a broadcast multiply-and-sum. Harmless for
ranking, but it made a descriptor's dot product with *itself* come out as
0.9994. Since that value feeds the ratio test, `matching.py` ranks with the
GEMM and then rescores the two survivors exactly — O(N) instead of O(N²).

## CudaSift's own tests: 29/29

The Pascal branch ships no test suite — no test files, no `add_test`, and
`mainSift.cpp` contains zero assertions. The only tests this codebase has ever
had were added in commit `6464e95` and **reverted by the maintainer in the very
next commit**, `b5ed6c2`, which is HEAD's direct parent. So they are not
authoritative upstream tests, but they are the only ones that exist, and they
are still in the object store:

```
git -C CudaSift show 6464e95:tests/test_extract.cpp
```

All 29 assertions from `test_extract.cpp` (10), `test_match.cpp` (11) and
`test_homography.cpp` (8) are reproduced verbatim in
`tests/test_cudasift_suite.py`, with the same images, transforms and
parameters. All 29 pass. Highlights:

| assertion | result |
|---|---|
| Higher threshold ⇒ fewer features | 7538 → 2061 → 588 → 6 for thresh 1/3/5/10 |
| More octaves ⇒ more or equal features | 1891 / 2027 / 2061 / 2070 for 3–6 octaves |
| Same features on repeated runs | 2061 = 2061, 100% position agreement |
| Scale-up mode detects more features | 3219 vs 2061 |
| Self-match >95% score > 0.95 | 2061/2061 |
| Translation (30, 20) within 5 px | recovered (30.25, 20.17), 1792 matches |
| 10° rotation, >15 matches | 1330 |
| 0.8 scale change, >10 matches | 1016 |
| Stereo pair left/righ.pgm, >20 matches | 1616 and 2268 features, 842 matches |
| Matching < 2 ms | 0.70 ms |

The reproducibility check is worth calling out, because the CAS-emulated
threadgroup float atomics accumulate in nondeterministic order and could in
principle flip a near-tied orientation histogram peak, changing the duplicate
count. In practice it is stable: identical counts and 100% position agreement
across runs.

One assertion — "all ambiguity values in [0, 1]" — genuinely failed at first
and caught a real bug: matching ranks with the fast GEMM but rescores the top
two exactly, and on a near-tie the exact rescore could invert the pair, pushing
the ratio just above 1. Fixed by re-ordering after the rescore.

## How correctness was otherwise established

There is no CUDA GPU here, so the port could not be diffed against CudaSift
directly. Three further substitutes, all in `tests/`:

1. **Scale-space vs. an independent NumPy reference** that reproduces the CUDA
   index arithmetic literally (clamped gathers, no convolution primitive).
   Agreement is at float32 rounding. An impulse test pins `ScaleDown`'s
   sampling to exactly `(2u, 2v)`, and `scale_up` matches the CUDA formula bit
   for bit.
2. **A synthetic warp with known ground truth** — 12° rotation, 0.85 scale,
   translation. Recovers 1002 RANSAC inliers with **0.56 px** corner
   reprojection error. This is the test that would fail loudly if the bilinear
   half-pixel convention were wrong, since a half-pixel bias survives matching
   but shows up in the geometry.
3. **OpenCV SIFT as a peer.** Under the same warp, detector repeatability is
   **0.698 vs OpenCV's 0.531**. On CudaSift's own image pair the recovered
   homography agrees with OpenCV's to 5.5 px at the image corners (118 fitted
   matches vs OpenCV's 123).

Positional overlap with OpenCV is only 0.58 at 2 px — but the two are
different detectors (5 scales per octave vs 3, different contrast and edge
tests, single-shot vs iterated subpixel refinement), and OpenCV finds 5303
keypoints to our 2061. The warp repeatability number above is the meaningful
comparison, and it favours this port.

The one thing still unverified is agreement with CudaSift *itself*. Keypoint
counts are 2061/2240 against the ~1911/2086 its README reports for this pair —
about 8% more, direction and cause not established without a CUDA machine to
diff against.
