"""End-to-end run mirroring mainSift.cpp: extract, match, fit a homography."""
import sys, time

import mlx.core as mx
import numpy as np

from metalsift.sift import extract_sift, load_gray
from metalsift import matching


def timed(fn, n=5):
    fn()                                  # warm up kernel compilation
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1e3)
    return min(ts), float(np.median(ts))


def main():
    imgset = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if imgset:
        f1, f2, thresh = "CudaSift/data/left.pgm", "CudaSift/data/righ.pgm", 4.5
    else:
        f1, f2, thresh = "CudaSift/data/img1.png", "CudaSift/data/img2.png", 3.0

    img1, img2 = load_gray(f1), load_gray(f2)
    print(f"Image size = ({img1.shape[1]}, {img1.shape[0]})")

    def run():
        r = extract_sift(img1, 5, 1.0, thresh)
        mx.eval(r["desc"])
        return r

    best, med = timed(run)
    s1 = extract_sift(img1, 5, 1.0, thresh)
    s2 = extract_sift(img2, 5, 1.0, thresh)
    print(f"SIFT extraction time = {best:.2f} ms (best of 5, median {med:.2f})")
    print(f"Number of original features: {s1['num']} {s2['num']}")

    t = time.perf_counter()
    m = matching.match(s1["desc"], s2["desc"], s2["xy"])
    mx.eval(m["score"])
    print(f"Matching time = {(time.perf_counter() - t) * 1e3:.2f} ms")

    H, n = matching.find_homography(s1["xy"], m, num_loops=10000,
                                    min_score=0.0, max_ambiguity=0.80, thresh=5.0)
    H, numfit = matching.improve_homography(s1["xy"], m, H, num_loops=5,
                                            min_score=0.0, max_ambiguity=0.80,
                                            thresh=3.0)
    score = np.asarray(m["score"])
    amb = np.asarray(m["ambiguity"])
    considered = int(((score > 0.0) & (amb < 0.80)).sum())
    print(f"Number of matching features: {numfit} {n} "
          f"{100.0 * numfit / max(considered, 1):.2f}% of {considered} considered")
    print("Homography:")
    print(np.array2string(H / H[2, 2], precision=4, suppress_small=True))


if __name__ == "__main__":
    main()
