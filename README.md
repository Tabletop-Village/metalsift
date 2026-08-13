# metalSIFT

A port of [CudaSift](https://github.com/Celebrandil/CudaSift) (Pascal branch, MIT)
to Apple Silicon via MLX + custom Metal kernels. Built and measured on an M5.

This is a derivative work of CudaSift, copyright (c) 2017 Mårten Björkman, and
is distributed under the same MIT licence with that copyright notice retained.
The algorithm, kernel structure and parameter choices are his; what is new here
is the Metal/MLX implementation.

## Install

```
pip install metalsift            # Apple Silicon only -- there is no CPU fallback
pip install 'metalsift[io]'      # adds OpenCV, needed only for load_gray()
```

```python
import metalsift, mlx.core as mx
kp1 = metalsift.extract_sift(img1)          # img: (H, W) float32 mlx array, 0-255
kp2 = metalsift.extract_sift(img2)
m = metalsift.match(kp1["desc"], kp2["desc"], kp2["xy"])
H, n = metalsift.find_homography(kp1["xy"], m)
```

The wheel is pure Python — the Metal shaders are strings that MLX compiles at
runtime — but it is tagged `macosx_11_0_arm64` so pip declines it on machines
with no Metal GPU rather than installing and failing later.

## Developing

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
| **full extraction** | **3.02** |
| ... excluding LowPass, CudaSift's benchmark basis | 2.63 |
| matching, 2061 x 2240 descriptors | 0.70 |

CudaSift reports 1.7 ms on a GTX 1060 and 0.80 ms on a 1080 Ti — but its
published figures exclude LowPass, which it treats as preprocessing, so 2.63 ms
is the comparable number.

Per stage, summed over all five octaves. Each is measured with its own sync, so
they total more than the pipeline itself — lazy evaluation overlaps them:

| stage | ms |
|---|---|
| LowPass | 0.39 |
| 4x ScaleDown | 0.22 |
| LaplaceMulti (DoG) | 1.94 |
| FindPointsMulti | 1.14 |
| ComputeOrientations | 0.74 |
| duplicate compaction | 0.78 |
| ExtractSiftDescriptors | 0.99 |

Everything above runs in hand-written Metal. Host-side orchestration in Python
costs **0.07 ms**, constant regardless of image size, so rewriting the driver in
C++ would recover about 1% — see "Why not C++" below.

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
single-channel 9-tap pass takes 2.4 ms. Building the DoG pyramid from `conv2d`
— the obvious pure-MLX approach — cost 38 of the original 45 ms.

Rewriting as fused shifted multiply-adds under `mx.compile` helps a lot, but
still writes a full-resolution intermediate between the row and column passes.
Every separable stage ended up worth a custom kernel:

| stage | conv2d | compiled shifts | Metal kernel |
|---|---|---|---|
| LaplaceMulti (8 scales) | 18.7 ms | 7.9 ms | **1.11 ms** |
| LowPass | 4.6 ms | 1.26 ms | **0.35 ms** |
| 4x ScaleDown | — | 1.24 ms | **0.19 ms** |

The ops versions are kept in `pyramid.py` as references, and the test suite
checks each kernel against them.

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

## Why not C++

Rewriting the host side in compiled code would buy almost nothing, because
there is no interpreted hot loop to remove. Every kernel is already compiled
Metal; Python only builds the dispatch graph. Timing the two separately:

| image | Mpix | Python build | GPU eval |
|---|---|---|---|
| 512x512 | 0.26 | 0.09 ms | 1.36 ms |
| 1024x1024 | 1.05 | 0.07 ms | 3.17 ms |
| 1920x1080 | 2.07 | 0.07 ms | 5.52 ms |

(measured before the LowPass/ScaleDown kernels landed). Python is flat at
0.07 ms; GPU time fits `2.28 ms/Mpix + 0.78 ms`. That 0.78 ms fixed cost covers
~25 kernel dispatches, which also shows MLX batches them into command buffers:
a single isolated kernel round trip costs 193 us, so 25 unbatched dispatches
would alone be 4.8 ms.

For streaming work it is better still — MLX dispatches asynchronously, so the
next frame's graph builds while the current one runs, hiding the 0.07 ms
entirely.

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
