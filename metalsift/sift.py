"""End-to-end SIFT extraction, mirroring ExtractSift / ExtractSiftLoop /
ExtractSiftOctave in cudaSiftH.cu.

Structural difference from CudaSift: instead of one global SiftPoint buffer
with a persistent d_PointCounter[17] mutated across octaves, each octave gets
its own buffer and counter. MLX kernel inputs are const, so the in-place
counter chain would need explicit threading anyway; separate buffers plus a
concatenate at the end is both simpler and costs exactly one device sync for
the whole extraction, at the very end.
"""
import mlx.core as mx
import numpy as np

from . import msl
from . import pyramid as P

NUM_SCALES = P.NUM_SCALES
EDGE_LIMIT = 10.0


def extract_sift(img, num_octaves=5, init_blur=1.0, thresh=3.0,
                 lowest_scale=0.0, scale_up=False, max_pts_per_octave=8192):
    """img: (H, W) float32 MLX array with 0-255 intensities."""
    if img.dtype != mx.float32:
        img = img.astype(mx.float32)

    if scale_up:
        img = P.scale_up(img)
        lowest_scale = lowest_scale * 2.0

    # Prefilter, then build the octave base images by successive ScaleDown.
    # ExtractSiftLoop recurses down before processing, so octave index 1 is the
    # smallest image and num_octaves is full resolution.
    base = msl.lowpass(img, mx.array(P.lowpass_taps(max(init_blur, 0.001))))
    sd_taps = mx.array(P.scaledown_taps(0.5))
    bases = {num_octaves: base}
    for o in range(num_octaves - 1, 0, -1):
        h, w = bases[o + 1].shape
        if w < 4 or h < 4:
            break
        bases[o] = msl.scale_down(bases[o + 1], sd_taps)

    lap = {o: mx.array(k) for o, k in P.laplace_taps(num_octaves, 0.0).items()}
    per_octave = []

    for o in range(1, num_octaves + 1):
        if o not in bases:
            continue
        oimg = bases[o]
        h, w = oimg.shape
        if w < 32 or h < 16:
            continue
        subsampling = float(2 ** (num_octaves - o))

        dog = msl.laplace_multi(oimg, lap[o], w, h)
        kp, cnt = msl.find_points(
            dog, w, h, max_pts_per_octave, subsampling,
            lowest_scale / subsampling, thresh, 1.0 / NUM_SCALES, EDGE_LIMIT)

        kp7, total = msl.compute_orientations(
            oimg, kp, cnt, w, h, max_pts_per_octave)

        desc = msl.descriptors(oimg, kp7, total, w, h, max_pts_per_octave)

        # ExtractSiftDescriptorsCONSTNew folds this rescale into the kernel;
        # as a pure op it avoids a second output buffer.
        mult = mx.array([subsampling, subsampling, subsampling, 1, 1, 1, 1],
                        dtype=mx.float32)
        per_octave.append((kp7 * mult, desc, total))

    # The one and only device sync of the whole extraction.
    mx.eval([t for _, _, t in per_octave])

    kps, descs = [], []
    for kp7, desc, total in per_octave:
        n = int(total.item())
        if n:
            kps.append(kp7[:n])
            descs.append(desc[:n])

    if not kps:
        return _empty()

    kp = mx.concatenate(kps, axis=0)
    desc = mx.concatenate(descs, axis=0)
    if scale_up:
        # RescalePositions(siftData, 0.5) -- cudaSiftD.cu:753
        kp = kp * mx.array([0.5, 0.5, 0.5, 1, 1, 1, 1], dtype=mx.float32)
    mx.eval(kp, desc)
    return {
        "xy": kp[:, 0:2],
        "scale": kp[:, 2],
        "sharpness": kp[:, 3],
        "edgeness": kp[:, 4],
        "subsampling": kp[:, 5],
        "orientation": kp[:, 6],
        "desc": desc,
        "num": kp.shape[0],
    }


def _empty():
    z = mx.zeros((0,), dtype=mx.float32)
    return {"xy": mx.zeros((0, 2), dtype=mx.float32), "scale": z,
            "sharpness": z, "edgeness": z, "subsampling": z,
            "orientation": z, "desc": mx.zeros((0, 128), dtype=mx.float32),
            "num": 0}


def load_gray(path):
    """Read an image as a (H, W) float32 MLX array with 0-255 intensities,
    matching mainSift.cpp's cv::imread(..., 0).convertTo(CV_32FC1).

    This is the only part of the package that needs OpenCV, which is why it is
    an optional dependency and imported here rather than at module scope.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "load_gray needs OpenCV, an optional dependency. Install it with: "
            "pip install 'metalsift[io]'  -- or decode the image yourself and "
            "pass a (H, W) float32 mlx array with 0-255 intensities to "
            "extract_sift()."
        ) from exc
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return mx.array(im.astype(np.float32))
