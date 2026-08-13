"""Scale-space construction: pure MLX ops, no custom Metal.

Mirrors the arithmetic of CudaSift's LowPassBlock, ScaleDown and
LaplaceMultiMem exactly (see cudaSiftD.cu). All filters are symmetric and
separable with clamp-to-edge boundary handling, which is what CUDA's pitched
loads with max(min(i, n-1), 0) index clamping amount to.
"""
import math

import mlx.core as mx
import numpy as np

NUM_SCALES = 5
LAPLACE_R = 4
LAPLACE_S = NUM_SCALES + 3   # 8 blurred scales -> 7 DoG planes
LOWPASS_R = 4


# --------------------------------------------------------------------------
# Host-side filter taps (PrepareLaplaceKernels / LowPass / ScaleDown)
# --------------------------------------------------------------------------

def lowpass_taps(scale):
    """9-tap Gaussian, cudaSiftH.cu:408-421."""
    j = np.arange(-LOWPASS_R, LOWPASS_R + 1, dtype=np.float64)
    k = np.exp(-j * j / (2.0 * scale * scale))
    return (k / k.sum()).astype(np.float32)


def scaledown_taps(variance=0.5):
    """5-tap Gaussian, cudaSiftH.cu:316-323."""
    j = np.arange(5, dtype=np.float64)
    k = np.exp(-(j - 2) ** 2 / 2.0 / variance)
    return (k / k.sum()).astype(np.float32)


def laplace_taps(num_octaves, init_blur=0.0):
    """Port of PrepareLaplaceKernels (cudaSiftH.cu:439).

    Returns {octave: (LAPLACE_S, 2*LAPLACE_R+1)} where octave indexes the way
    ExtractSiftOctave does: 1 = smallest image, num_octaves = full resolution.
    The recursion halves the accumulated blur each time it steps down an
    octave, matching ExtractSiftLoop's totInitBlur.
    """
    out = {}

    def rec(n, blur):
        if n > 1:
            rec(n - 1, math.sqrt(blur * blur + 0.25) / 2.0)
        half = np.zeros((LAPLACE_S, LAPLACE_R + 1), dtype=np.float64)
        scale = 2.0 ** (-1.0 / NUM_SCALES)
        diff_scale = 2.0 ** (1.0 / NUM_SCALES)
        for i in range(LAPLACE_S):
            var = scale * scale - blur * blur
            j = np.arange(LAPLACE_R + 1, dtype=np.float64)
            row = np.exp(-j * j / 2.0 / var)
            # weight of 1 for the centre tap, 2 for each mirrored pair
            half[i] = row / (row[0] + 2.0 * row[1:].sum())
            scale *= diff_scale
        # mirror the half-kernel out to the full symmetric 9 taps
        out[n] = np.concatenate([half[:, :0:-1], half], axis=1).astype(np.float32)

    rec(num_octaves, init_blur)
    return out


# --------------------------------------------------------------------------
# Separable convolution helpers
# --------------------------------------------------------------------------

def _pad_edge(x, py, px):
    return mx.pad(x, [(0, 0), (py, py), (px, px), (0, 0)], mode="edge")


def separable_conv(img, taps, stride=1):
    """Reference formulation via mx.conv2d. Kept because it is the obvious
    reading of the algorithm and the compiled version below is checked against
    it, but it is not used in the pipeline: MLX's conv2d is slow at these
    shapes (1 input channel, 9-wide kernel) -- 2.4 ms per pass at 1920x1080,
    versus 1.3 ms for the whole separable filter as fused shifts."""
    r = (len(taps) - 1) // 2
    x = img[None, :, :, None]
    wx = mx.array(taps.reshape(1, 1, -1, 1))
    x = mx.conv2d(_pad_edge(x, 0, r), wx, stride=(1, stride))
    wy = mx.array(taps.reshape(1, -1, 1, 1))
    x = mx.conv2d(_pad_edge(x, r, 0), wy, stride=(stride, 1))
    return x[0, :, :, 0]


def _make_separable(r, stride):
    """Symmetric separable filter as shifted multiply-adds, fused by
    mx.compile into a single kernel per pass. Symmetry halves the multiplies:
    k[c]*x[i] + sum_j k[c+j]*(x[i-j] + x[i+j])."""
    @mx.compile
    def f(img, k):
        h, w = img.shape
        ow, oh = w // stride, h // stride
        xp = mx.pad(img, [(0, 0), (r, r)], mode="edge")
        o = k[r] * xp[:, r:r + stride * ow:stride]
        for j in range(1, r + 1):
            o = o + k[r + j] * (xp[:, r - j:r - j + stride * ow:stride]
                                + xp[:, r + j:r + j + stride * ow:stride])
        yp = mx.pad(o, [(r, r), (0, 0)], mode="edge")
        out = k[r] * yp[r:r + stride * oh:stride]
        for j in range(1, r + 1):
            out = out + k[r + j] * (yp[r - j:r - j + stride * oh:stride]
                                    + yp[r + j:r + j + stride * oh:stride])
        return out
    return f


_sep_lowpass = _make_separable(LOWPASS_R, 1)
_sep_scaledown = _make_separable(2, 2)


def separable(img, taps, stride=1):
    f = _sep_lowpass if stride == 1 else _sep_scaledown
    return f(img, mx.array(taps))


def lowpass(img, scale):
    """LowPassBlock (cudaSiftD.cu:1986) as MLX ops.

    Reference implementation; msl.lowpass is what the pipeline uses. Two
    compiled passes cost 1.26 ms at 1920x1080 against 0.35 ms fused, because
    the row pass writes a full-resolution intermediate the column pass reads
    straight back.
    """
    return separable(img, lowpass_taps(scale))


def scale_down(img, taps):
    """ScaleDown (cudaSiftD.cu:84).

    Output pixel (u, v) is the 5-tap response centred on input (2u, 2v), which
    is what the shared-memory version computes; there is no half-pixel shift.
    Trailing odd row/column is dropped, as CudaSift does via width/2.

    Reference implementation; msl.scale_down is what the pipeline uses
    (1.24 ms vs 0.19 ms for the four pyramid levels at 1920x1080).
    """
    return separable(img, taps, stride=2)


def scale_up(img):
    """ScaleUp (cudaSiftD.cu:170).

    Not mlx.nn.Upsample(mode="linear"): the CUDA kernel puts the source
    samples on the *even* output pixels and averages neighbours into the odd
    ones, with edge clamp. MLX's linear upsample uses a different phase under
    either align_corners setting, which would shift every keypoint.
    """
    xr = mx.concatenate([img[:, 1:], img[:, -1:]], axis=1)      # x+1, clamped
    yd = mx.concatenate([img[1:, :], img[-1:, :]], axis=0)      # y+1, clamped
    xyd = mx.concatenate([xr[1:, :], xr[-1:, :]], axis=0)
    q = mx.stack([
        mx.stack([img, 0.5 * (img + xr)], axis=-1),
        mx.stack([0.5 * (img + yd), 0.25 * (img + xr + yd + xyd)], axis=-1),
    ], axis=-2)                                                 # (h, w, 2, 2)
    h, w = img.shape
    return mx.transpose(q, (0, 2, 1, 3)).reshape(2 * h, 2 * w)


def laplace_multi_ops(img, taps):
    """LaplaceMultiMem (cudaSiftD.cu:1753) as pure MLX ops.

    Reference implementation only -- msl.laplace_multi is what the pipeline
    uses. This version materialises all 8 blurred planes (66 MB at 1920x1080)
    before differencing them, and mx.conv2d's depthwise path costs 18.7 ms on
    its own. Retained so the fused Metal kernel has something to be checked
    against. taps: (LAPLACE_S, 9). Returns (LAPLACE_S-1, H, W).
    """
    r = (taps.shape[1] - 1) // 2
    n = taps.shape[0]
    x = img[None, :, :, None]
    # row pass: one output channel per scale, all from the single input channel
    wx = mx.array(taps.reshape(n, 1, -1, 1))
    x = mx.conv2d(_pad_edge(x, 0, r), wx)
    # column pass: depthwise, each channel keeps its own kernel
    wy = mx.array(taps.reshape(n, -1, 1, 1))
    x = mx.conv2d(_pad_edge(x, r, 0), wy, groups=n)
    blur = x[0]                                   # (H, W, LAPLACE_S)
    dog = blur[:, :, 1:] - blur[:, :, :-1]        # (H, W, LAPLACE_S-1)
    return mx.transpose(dog, (2, 0, 1))           # (7, H, W), contiguous
