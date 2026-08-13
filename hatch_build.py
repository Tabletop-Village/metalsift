"""Tag the wheel for macOS arm64.

metalsift contains no compiled code -- the Metal shaders are Python strings
that MLX compiles at runtime -- so hatchling would build a `py3-none-any`
wheel. But mlx installs happily on Linux (it pulls mlx-metal only on Darwin),
so an `any` wheel would let `pip install metalsift` succeed on a machine with
no Metal GPU and then fail deep inside kernel compilation. A platform tag makes
pip decline it up front instead.

The sdist has no such tag, so pip can still fall back to building from source
on an unsupported platform; the import-time check in metalsift/__init__.py
catches that case.
"""
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PLAT_TAG = "macosx_11_0_arm64"


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        build_data["pure_python"] = False
        build_data["infer_tag"] = False
        build_data["tag"] = f"py3-none-{PLAT_TAG}"
