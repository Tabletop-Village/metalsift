"""SIFT feature extraction on Apple Silicon.

A port of CudaSift (https://github.com/Celebrandil/CudaSift, MIT, copyright
(c) 2017 Marten Bjorkman) to MLX with custom Metal kernels.

    import metalsift
    kp = metalsift.extract_sift(img)             # img: (H, W) float32, 0-255
    m  = metalsift.match(kp["desc"], kp2["desc"], kp2["xy"])
"""
__version__ = "0.1.0"


def _require_metal():
    """Fail with something readable rather than an opaque shader-compile error.

    Every kernel here needs a Metal GPU. The wheel is tagged for macOS arm64 so
    pip normally declines to install it elsewhere, but an sdist build or a
    source checkout can still get this far on an unsupported machine.
    """
    import platform

    try:
        import mlx.core as mx
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "metalsift requires mlx, which is not installed. "
            "Install it with: pip install metalsift"
        ) from exc

    available = getattr(getattr(mx, "metal", None), "is_available", None)
    if available is not None and not available():
        raise ImportError(
            "metalsift requires a Metal GPU (Apple Silicon). mlx reports no "
            f"Metal device on {platform.system()}/{platform.machine()}. "
            "There is no CPU fallback: every kernel is hand-written MSL."
        )


_require_metal()

from .matching import find_homography, improve_homography, match  # noqa: E402
from .sift import extract_sift, load_gray  # noqa: E402
from . import msl, pyramid  # noqa: E402

__all__ = [
    "__version__",
    "extract_sift",
    "load_gray",
    "match",
    "find_homography",
    "improve_homography",
    "msl",
    "pyramid",
]
