"""CudaSift's own test suite, ported.

The Pascal branch ships no tests. The only test suite this codebase has ever
had was added in commit 6464e95 ("add tests/demos/benchmarks") and reverted the
next commit by the maintainer, so these are not authoritative upstream tests --
but they are the only ones that exist, and every assertion is reproduced here
verbatim from tests/test_extract.cpp (10), tests/test_match.cpp (11) and
tests/test_homography.cpp (8) at that commit.

Recover the originals with:
    git -C CudaSift show 6464e95:tests/test_extract.cpp
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mlx.core as mx
import numpy as np

from metalsift import matching
from metalsift.sift import extract_sift

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "CudaSift", "data")
passed, failed = 0, 0


def CHECK(cond, msg):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {msg}")
    else:
        failed += 1
        print(f"  [FAIL] {msg}")


def load(name):
    im = cv2.imread(os.path.join(DATA, name), 0)
    return im


def sift(im, octaves=5, blur=1.0, thresh=3.0, scale_up=False):
    return extract_sift(mx.array(im.astype(np.float32)), octaves, blur, thresh,
                        0.0, scale_up)


def pipeline(s1, s2, num_loops=10000, min_score=0.0, max_amb=0.80,
             ransac_thresh=5.0, improve=True):
    m = matching.match(s1["desc"], s2["desc"], s2["xy"])
    H, n = matching.find_homography(s1["xy"], m, num_loops, min_score,
                                    max_amb, ransac_thresh)
    numfit = 0
    if improve:
        H, numfit = matching.improve_homography(s1["xy"], m, H, 5, min_score,
                                                max_amb, 3.0)
    return m, H, n, numfit


# ---------------------------------------------------------------------------
# test_extract.cpp
# ---------------------------------------------------------------------------

def test_basic_extraction():
    print("\n--- Test: Basic Extraction ---")
    img = load("img1.png")
    CHECK(img is not None, "Image loaded successfully")
    s = sift(img)
    CHECK(s["num"] > 0, "Features detected (numPts > 0)")
    CHECK(s["num"] < 32768, "Features within capacity")

    xy = np.asarray(s["xy"])[:100]
    sc = np.asarray(s["scale"])[:100]
    o = np.asarray(s["orientation"])[:100]
    h, w = img.shape
    valid = ((xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
             & (sc > 0) & ~np.isnan(sc) & (o >= 0) & (o < 360)).all()
    CHECK(bool(valid), "All feature positions/scales/orientations are valid")

    norms = np.linalg.norm(np.asarray(s["desc"])[:50], axis=1)
    CHECK(bool(((norms >= 0.8) & (norms <= 1.2)).all()),
          "Descriptor vectors are approximately unit length")


def test_different_thresholds():
    print("\n--- Test: Threshold Sensitivity ---")
    img = load("img1.png")
    prev, monotonic = 99999, True
    for t in (1.0, 3.0, 5.0, 10.0):
        n = sift(img, thresh=t)["num"]
        print(f"    thresh={t} -> {n} features")
        if n > prev:
            monotonic = False
        prev = n
    CHECK(monotonic, "Higher threshold = fewer features (monotonic decrease)")


def test_different_octaves():
    print("\n--- Test: Octave Count ---")
    img = load("img1.png")
    counts = []
    for oct_ in range(3, 7):
        n = sift(img, octaves=oct_)["num"]
        counts.append(n)
        print(f"    octaves={oct_} -> {n} features")
    CHECK(counts[1] >= counts[0], "More octaves = more or equal features")


def test_reproducibility():
    print("\n--- Test: Reproducibility ---")
    img = load("img1.png")
    a, b = sift(img), sift(img)
    CHECK(a["num"] == b["num"], "Same number of features on repeated runs")
    if a["num"] == b["num"] and a["num"] > 0:
        def srt(s):
            p = np.asarray(s["xy"])
            return p[np.lexsort((p[:, 0], p[:, 1]))]
        p1, p2 = srt(a), srt(b)
        n = int((np.abs(p1 - p2) < 0.1).all(axis=1).sum())
        ratio = n / a["num"]
        print(f"    Matching positions (sorted): {n}/{a['num']} ({ratio*100:.2f}%)")
        CHECK(ratio > 0.95, "Feature positions are consistent across runs (>95%)")


def test_scale_up():
    print("\n--- Test: Scale-Up Mode ---")
    img = load("img1.png")
    normal = sift(img)["num"]
    up = sift(img, scale_up=True)["num"]
    print(f"    Normal: {normal} features")
    print(f"    ScaleUp: {up} features")
    CHECK(up > normal, "Scale-up mode detects more features")


# ---------------------------------------------------------------------------
# test_match.cpp
# ---------------------------------------------------------------------------

def test_self_match():
    print("\n--- Test: Self-Matching ---")
    img = load("img1.png")
    s1, s2 = sift(img), sift(img)
    m = matching.match(s1["desc"], s2["desc"], s2["xy"])
    score = np.asarray(m["score"])
    ratio = float((score > 0.95).mean())
    print(f"    High score matches (>0.95): {int((score > 0.95).sum())}/{s1['num']}")
    CHECK(ratio > 0.95, "Self-match: >95% high score matches (>0.95)")
    CHECK(ratio > 0.80, "Self-match: >80% high score matches (sanity check)")


def test_match_quality():
    print("\n--- Test: Match Quality ---")
    s1, s2 = sift(load("img1.png")), sift(load("img2.png"))
    print(f"    Features: {s1['num']} vs {s2['num']}")
    m = matching.match(s1["desc"], s2["desc"], s2["xy"])
    idx = np.asarray(m["match"])
    score = np.asarray(m["score"])
    amb = np.asarray(m["ambiguity"])
    valid = int(((idx >= 0) & (idx < s2["num"])).sum())
    good = int((score > 0.7).sum())
    print(f"    Valid matches: {valid}")
    print(f"    Good matches (score > 0.7): {good}")
    CHECK(valid == s1["num"], "All matches have valid indices")
    CHECK(good > 50, "At least 50 good matches found")
    CHECK(bool(((amb >= 0.0) & (amb <= 1.0)).all()),
          "All ambiguity values in [0, 1]")


def test_homography_estimation():
    print("\n--- Test: Homography Estimation ---")
    s1, s2 = sift(load("img1.png")), sift(load("img2.png"))
    m, H, n, numfit = pipeline(s1, s2)
    print(f"    RANSAC matches: {n}")
    print(f"    Refined inliers: {numfit}")
    CHECK(n > 20, "RANSAC found > 20 matches")
    CHECK(numfit > 10, "Refined homography has > 10 inliers")

    h = (H / H[2, 2]).reshape(-1)
    not_identity = any(
        abs(h[i] - 1.0) > 0.01 if i in (0, 4) else abs(h[i]) > 0.01
        for i in range(8))
    CHECK(not_identity, "Homography is not identity (images differ)")
    CHECK(int((m["match_error"] < 5.0).sum()) > 0,
          "Some matches have low reprojection error")


def test_matching_performance():
    print("\n--- Test: Matching Performance ---")
    s1, s2 = sift(load("img1.png")), sift(load("img2.png"))
    for _ in range(10):
        mx.eval(matching.match(s1["desc"], s2["desc"], s2["xy"])["score"])
    t0 = time.perf_counter()
    runs = 100
    for _ in range(runs):
        mx.eval(matching.match(s1["desc"], s2["desc"], s2["xy"])["score"])
    avg = (time.perf_counter() - t0) * 1e3 / runs
    print(f"    Features: {s1['num']} x {s2['num']}")
    print(f"    Avg matching time: {avg:.3f} ms")
    CHECK(avg < 5.0, "Matching completes in < 5 ms")
    CHECK(avg < 2.0, "Matching completes in < 2 ms (optimal)")


# ---------------------------------------------------------------------------
# test_homography.cpp
# ---------------------------------------------------------------------------

def test_translation():
    print("\n--- Test: Translation Detection ---")
    img = load("img1.png")
    M = np.float32([[1, 0, 30], [0, 1, 20]])
    trans = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    s1, s2 = sift(img), sift(trans)
    m, H, n, _ = pipeline(s1, s2)
    h = (H / H[2, 2]).reshape(-1)
    print("    Expected translation: (30, 20)")
    print(f"    Recovered h[2]={h[2]:.4f} h[5]={h[5]:.4f}")
    print(f"    Matches: {n}")
    CHECK(n > 20, "Found > 20 RANSAC matches for translation")
    CHECK(abs(h[2] - 30.0) < 5.0, "Translation X recovered within 5 pixels")
    CHECK(abs(h[5] - 20.0) < 5.0, "Translation Y recovered within 5 pixels")


def test_rotation():
    print("\n--- Test: Rotation Detection ---")
    img = load("img1.png")
    h, w = img.shape
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 10.0, 1.0)
    rotated = cv2.warpAffine(img, rot, (w, h))
    s1, s2 = sift(img), sift(rotated)
    _, _, n, _ = pipeline(s1, s2, improve=False)
    print("    Rotation: 10 degrees")
    print(f"    Matches found: {n}")
    CHECK(n > 15, "Found > 15 RANSAC matches for rotation")


def test_scale():
    print("\n--- Test: Scale Change Detection ---")
    img = load("img1.png")
    scaled = cv2.resize(img, None, fx=0.8, fy=0.8)
    padded = np.zeros_like(img)
    padded[:scaled.shape[0], :scaled.shape[1]] = scaled
    s1, s2 = sift(img), sift(padded)
    _, _, n, _ = pipeline(s1, s2, improve=False)
    print("    Scale factor: 0.8")
    print(f"    Matches found: {n}")
    CHECK(n > 10, "Found > 10 RANSAC matches for scale change")


def test_stereo_pair():
    print("\n--- Test: Stereo Pair (left/righ.pgm) ---")
    limg, rimg = load("left.pgm"), load("righ.pgm")
    if limg is None or rimg is None:
        print("  [SKIP] PGM images not found")
        return
    s1, s2 = sift(limg, thresh=4.5), sift(rimg, thresh=4.5)
    print(f"    Left features: {s1['num']}")
    print(f"    Right features: {s2['num']}")
    _, _, n, numfit = pipeline(s1, s2)
    print(f"    RANSAC matches: {n}")
    print(f"    Inliers: {numfit}")
    CHECK(s1["num"] > 100, "Left image has > 100 features")
    CHECK(s2["num"] > 100, "Right image has > 100 features")
    CHECK(n > 20, "Found > 20 matches between stereo pair")


def main():
    if not os.path.isdir(DATA):
        print("SKIP: CudaSift test images not found.\n"
              "      Run ./scripts/fetch_cudasift.sh to fetch them.")
        return 0

    print("=== test_extract.cpp (10 checks) ===")
    test_basic_extraction()
    test_different_thresholds()
    test_different_octaves()
    test_reproducibility()
    test_scale_up()

    print("\n=== test_match.cpp (11 checks) ===")
    test_self_match()
    test_match_quality()
    test_homography_estimation()
    test_matching_performance()

    print("\n=== test_homography.cpp (8 checks) ===")
    test_translation()
    test_rotation()
    test_scale()
    test_stereo_pair()

    print(f"\n{'=' * 46}\nPassed: {passed}   Failed: {failed}   "
          f"Total: {passed + failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
