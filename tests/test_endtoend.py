"""End-to-end validation of the Metal SIFT port.

There is no CUDA reference available on this machine, so correctness is
established three ways:

  1. invariants that must hold for any correct SIFT implementation;
  2. a synthetic warp with a known ground-truth homography -- this is what
     actually catches a wrong bilinear half-pixel convention or a broken
     descriptor, since both destroy match geometry;
  3. agreement with OpenCV SIFT on the repository's own test images.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mlx.core as mx
import numpy as np

from metalsift.sift import extract_sift, load_gray
from metalsift import matching

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "CudaSift", "data")
fails = []


def report(name, ok, detail=""):
    print(f"  {name:<44} {'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        fails.append(name)


def test_invariants(s):
    print("invariants")
    d = np.asarray(s["desc"])
    norms = np.linalg.norm(d, axis=1)
    report("descriptors L2-normalised", np.abs(norms - 1).max() < 1e-5,
           f"max dev {np.abs(norms - 1).max():.2e}")
    report("descriptors finite and non-negative",
           np.isfinite(d).all() and (d >= 0).all())
    report("descriptor bins clipped at 0.2",
           d.max() <= 0.2 / np.sqrt(0.2) + 1e-3, f"max {d.max():.4f}")
    o = np.asarray(s["orientation"])
    report("orientations in [0, 360)", (o >= 0).all() and (o < 360).all(),
           f"[{o.min():.2f}, {o.max():.2f}]")
    e = np.asarray(s["edgeness"])
    # tra^2/det >= 4 always, and the kernel rejects >= edgeLimit (10)
    report("edgeness within [4, 10)", e.min() >= 4 - 1e-3 and e.max() < 10,
           f"[{e.min():.3f}, {e.max():.3f}]")
    xy = np.asarray(s["xy"])
    report("keypoints inside the image",
           (xy[:, 0] >= 0).all() and (xy[:, 0] < 1920).all()
           and (xy[:, 1] >= 0).all() and (xy[:, 1] < 1080).all())


def test_self_match(s):
    print("self-consistency")
    m = matching.match(s["desc"], s["desc"], s["xy"])
    sc = np.asarray(m["score"])
    report("self-match score == 1", np.abs(sc - 1).max() < 1e-5,
           f"min {sc.min():.6f}")
    idx = np.asarray(m["match"])
    # ties are possible where two keypoints share a location; only require
    # that the winner's descriptor is identical to the query's
    d = np.asarray(s["desc"])
    same = np.isclose(d, d[idx], atol=1e-6).all(axis=1)
    report("self-match returns an identical descriptor", same.all(),
           f"{(~same).sum()} exceptions")


def test_synthetic_warp(img):
    """Warp by a known homography and try to recover it. This is the test that
    fails loudly if the bilinear half-pixel convention is wrong."""
    print("synthetic warp recovery (rot 12 deg, scale 0.85, translation)")
    h, w = img.shape
    th = np.deg2rad(12.0)
    s = 0.85
    Hgt = np.array([
        [s * np.cos(th), -s * np.sin(th), 130.0],
        [s * np.sin(th),  s * np.cos(th),  70.0],
        [0.0, 0.0, 1.0]])
    src = np.asarray(img).astype(np.float32)
    warped = cv2.warpPerspective(src, Hgt, (w, h), flags=cv2.INTER_LINEAR)

    s1 = extract_sift(mx.array(src), 5, 1.0, 3.0)
    s2 = extract_sift(mx.array(warped), 5, 1.0, 3.0)
    m = matching.match(s1["desc"], s2["desc"], s2["xy"])
    H, ninl = matching.find_homography(s1["xy"], m, num_loops=4000,
                                       max_ambiguity=0.90, thresh=3.0)
    H, nfit = matching.improve_homography(s1["xy"], m, H, num_loops=5,
                                          max_ambiguity=0.90, thresh=3.0)

    # compare by how far the two homographies move the image corners
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float).T
    def proj(M):
        p = M @ corners
        return (p[:2] / p[2]).T
    err = np.linalg.norm(proj(H) - proj(Hgt), axis=1).max()

    report("recovers >=300 RANSAC inliers", ninl >= 300, f"{ninl} inliers")
    report("corner reprojection error < 1.5 px", err < 1.5, f"{err:.3f} px")

    # Detector repeatability: fraction of in-view keypoints that reappear at
    # the mapped location. Measured against OpenCV as a peer, since there is
    # no CudaSift reference on this machine to diff against directly.
    def repeatability(p1, p2, tol=2.0):
        q = Hgt @ np.column_stack([p1, np.ones(len(p1))]).T
        q = (q[:2] / q[2]).T
        q = q[(q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)]
        return float((np.linalg.norm(q[:, None, :] - p2[None, :, :],
                                     axis=2).min(1) < tol).mean())

    ours = repeatability(np.asarray(s1["xy"]), np.asarray(s2["xy"]))
    sift = cv2.SIFT_create()
    u8 = src.astype(np.uint8)
    cv_rep = repeatability(
        np.array([k.pt for k in sift.detect(u8, None)]),
        np.array([k.pt for k in sift.detect(warped.astype(np.uint8), None)]))
    report("detector repeatability >= OpenCV's", ours >= cv_rep,
           f"ours {ours:.3f} vs OpenCV {cv_rep:.3f}")
    return err


def test_vs_opencv(s1, s2, im1, im2):
    print("agreement with OpenCV SIFT")
    sift = cv2.SIFT_create()
    k1 = sift.detect(im1, None)
    cv_pts = np.array([k.pt for k in k1])
    ours = np.asarray(s1["xy"])

    # Positional overlap with OpenCV. Informational only: CudaSift uses 5
    # scales per octave to OpenCV's 3, a different contrast threshold scale,
    # a different edge test and a single-shot subpixel refit, so the two
    # detectors are not expected to agree keypoint-for-keypoint.
    d = np.linalg.norm(ours[:, None, :] - cv_pts[None, :, :], axis=2)
    nearest = d.min(axis=1)
    print(f"  {'(info) OpenCV positional overlap':<44}        "
          f"{(nearest < 2.0).mean():.3f} @2px, {(nearest < 4.0).mean():.3f} @4px "
          f"({len(ours)} ours vs {len(cv_pts)} OpenCV)")

    # matching outcome on the same pair
    m = matching.match(s1["desc"], s2["desc"], s2["xy"])
    H, ninl = matching.find_homography(s1["xy"], m, num_loops=10000,
                                       max_ambiguity=0.80, thresh=5.0)
    H, nfit = matching.improve_homography(s1["xy"], m, H, thresh=3.0)

    kk1, dd1 = sift.detectAndCompute(im1, None)
    kk2, dd2 = sift.detectAndCompute(im2, None)
    dd1 = dd1 / np.linalg.norm(dd1, axis=1, keepdims=True)
    dd2 = dd2 / np.linalg.norm(dd2, axis=1, keepdims=True)
    c = dd1 @ dd2.T
    i = np.argmax(c, axis=1)
    best = c[np.arange(len(c)), i]
    c[np.arange(len(c)), i] = -1
    amb = c.max(1) / (best + 1e-6)
    p1 = np.array([k.pt for k in kk1])
    p2 = np.array([k.pt for k in kk2])
    sel = amb < 0.8
    Hcv, mask = cv2.findHomography(p1[sel], p2[i[sel]], cv2.RANSAC, 3.0)

    corners = np.array([[0, 0, 1], [1920, 0, 1], [0, 1080, 1], [1920, 1080, 1]],
                       float).T
    def proj(M):
        p = M @ corners
        return (p[:2] / p[2]).T
    err = np.linalg.norm(proj(H / H[2, 2]) - proj(Hcv / Hcv[2, 2]), axis=1).max()
    report("homography agrees with OpenCV within 8 px", err < 8.0,
           f"{err:.2f} px corner disagreement; ours {nfit} fit / {ninl} inliers, "
           f"OpenCV {int(mask.sum())} inliers")


def main():
    if not os.path.isdir(DATA):
        print("SKIP: CudaSift test images not found.\n"
              "      Run ./scripts/fetch_cudasift.sh to fetch them.")
        return 0

    img1 = load_gray(os.path.join(DATA, "img1.png"))
    img2 = load_gray(os.path.join(DATA, "img2.png"))
    im1 = cv2.imread(os.path.join(DATA, "img1.png"), 0)
    im2 = cv2.imread(os.path.join(DATA, "img2.png"), 0)

    s1 = extract_sift(img1, 5, 1.0, 3.0)
    s2 = extract_sift(img2, 5, 1.0, 3.0)
    print(f"extracted {s1['num']} and {s2['num']} keypoints\n")

    test_invariants(s1)
    test_self_match(s1)
    test_synthetic_warp(img1)
    test_vs_opencv(s1, s2, im1, im2)

    print("\nFAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
