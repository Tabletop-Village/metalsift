"""Descriptor matching and homography estimation, in pure MLX ops.

CudaSift's FindMaxCorr10 (matching.cu:301) is 120 lines of hand-tiled
shared-memory blocking and register accumulation. Since descriptors are
L2-normalised, correlation is just a dot product, so the whole thing is one
GEMM plus a best/second-best reduction -- which inherits MLX's tuned matmul.

Two behavioural notes versus the original:
  * FindMaxCorr10 iterates `bp2 < numPts2 - 32 + 1` and so silently ignores up
    to 31 trailing descriptors of set 2. This does not.
  * RANSAC hypothesis sampling uses NumPy's RNG rather than rand(), so inlier
    counts vary run to run by a few, same as the original does across seeds.
"""
import mlx.core as mx
import numpy as np


def match(desc1, desc2, xy2):
    """Best and second-best correlation for every descriptor in set 1.

    Returns dict with score, ambiguity, match index, and matched coordinates,
    matching the fields FindMaxCorr10 writes back into SiftPoint.
    """
    corr = mx.matmul(desc1, desc2.T)                      # (N1, N2)
    idx = mx.argmax(corr, axis=1)
    # mask out the winner to find the runner-up
    masked = mx.where(
        mx.arange(corr.shape[1])[None, :] == idx[:, None], -1e30, corr)
    idx2 = mx.argmax(masked, axis=1)

    # MLX's float32 GEMM dispatches to a reduced-precision hardware path on
    # Apple silicon (~8e-4 relative error, vs 4e-7 for the CPU path). That is
    # fine for ranking but not for the score itself, which feeds the ratio
    # test, so rescore the two survivors exactly -- elementwise mul + sum
    # keeps full float32 and costs O(N) instead of O(N^2).
    a = mx.sum(desc1 * mx.take(desc2, idx, axis=0), axis=1)
    b = mx.sum(desc1 * mx.take(desc2, idx2, axis=0), axis=1)
    # Ranking came from the approximate GEMM, so an exact rescore can invert a
    # near-tie. Re-order the pair, or ambiguity creeps just above 1.
    swap = b > a
    score = mx.where(swap, b, a)
    second = mx.where(swap, a, b)
    idx = mx.where(swap, idx2, idx)
    return {
        "score": score,
        "ambiguity": second / (score + 1e-6),
        "match": idx,
        "match_xy": mx.take(xy2, idx, axis=0),
    }


def find_homography(xy1, m, num_loops=10000, min_score=0.0,
                    max_ambiguity=0.80, thresh=5.0, seed=0):
    """4-point DLT RANSAC, vectorised over hypotheses.

    Port of FindHomography + ComputeHomographies + TestHomographies
    (matching.cu:910-1000). Returns (homography 3x3, inlier count).
    """
    score = np.asarray(m["score"])
    amb = np.asarray(m["ambiguity"])
    p1 = np.asarray(xy1, np.float64)
    p2 = np.asarray(m["match_xy"], np.float64)

    valid = np.flatnonzero((score > min_score) & (amb < max_ambiguity))
    if valid.size < 8:
        return np.eye(3), 0
    # Orientation doubling emits several keypoints at one position, so sample
    # hypotheses from distinct positions only -- drawing two correspondences
    # from the same point gives duplicate rows and an exactly singular system.
    _, uniq = np.unique(p1[valid].round(3), axis=0, return_index=True)
    valid = valid[np.sort(uniq)]
    if valid.size < 8:
        return np.eye(3), 0
    v1, v2 = p1[valid], p2[valid]

    rng = np.random.default_rng(seed)
    # four distinct correspondences per hypothesis
    sel = np.argsort(rng.random((num_loops, valid.size)), axis=1)[:, :4]
    x1, y1 = v1[sel, 0], v1[sel, 1]                       # (L, 4)
    x2, y2 = v2[sel, 0], v2[sel, 1]

    # Build the 8x8 system exactly as ComputeHomographies does
    L = num_loops
    A = np.zeros((L, 8, 8))
    b = np.zeros((L, 8))
    z = np.zeros_like(x1)
    o = np.ones_like(x1)
    A[:, 0::2, :] = np.stack([x1, y1, o, z, z, z, -x2 * x1, -x2 * y1], -1)
    A[:, 1::2, :] = np.stack([z, z, z, x1, y1, o, -y2 * x1, -y2 * y1], -1)
    b[:, 0::2] = x2
    b[:, 1::2] = y2

    # Collinear samples still make A singular, so use pinv rather than solve:
    # a degenerate hypothesis then yields a min-norm solution that simply
    # loses the inlier vote, instead of raising.
    h = (np.linalg.pinv(A) @ b[..., None])[..., 0]
    h[~np.isfinite(h).all(axis=1)] = 0.0

    # Inlier counting on GPU: one matmul over all hypotheses at once
    inl = _count_inliers(h, p1, p2, thresh)
    best = int(np.argmax(inl))
    H = np.append(h[best], 1.0).reshape(3, 3)
    return H, int(inl[best])


def _count_inliers(h, p1, p2, thresh):
    """Vectorised TestHomographies: (L, 8) hypotheses vs. N correspondences."""
    hm = mx.array(h.astype(np.float32))
    pts = mx.array(np.column_stack([p1, np.ones(len(p1))]).astype(np.float32))
    x2 = mx.array(p2[:, 0].astype(np.float32))
    y2 = mx.array(p2[:, 1].astype(np.float32))

    nomx = mx.matmul(hm[:, 0:3], pts.T)                   # (L, N)
    nomy = mx.matmul(hm[:, 3:6], pts.T)
    deno = mx.matmul(hm[:, 6:8], pts[:, 0:2].T) + 1.0
    errx = x2[None, :] * deno - nomx
    erry = y2[None, :] * deno - nomy
    err2 = errx * errx + erry * erry
    hit = err2 < (thresh * thresh) * deno * deno
    return np.asarray(mx.sum(hit.astype(mx.int32), axis=1))


def improve_homography(xy1, m, H, num_loops=5, min_score=0.0,
                       max_ambiguity=0.80, thresh=3.0):
    """Port of ImproveHomography (geomFuncs.cpp:6): iteratively reweighted
    least squares over the current inlier set.

    Returns (H, num_fit). Also writes m["match_error"] for *every*
    correspondence, not just the retained ones, as the original does in its
    final loop over numPts.
    """
    score = np.asarray(m["score"])
    amb = np.asarray(m["ambiguity"])
    keep = (score >= min_score) & (amb <= max_ambiguity)
    all1 = np.asarray(xy1, np.float64)
    all2 = np.asarray(m["match_xy"], np.float64)
    p1, p2 = all1[keep], all2[keep]
    if len(p1) < 8:
        m["match_error"] = np.full(len(all1), np.inf)
        return H, 0

    limit = thresh * thresh
    A = (H.reshape(-1)[:8] / H.reshape(-1)[8]).copy()
    x, y = p1[:, 0], p1[:, 1]
    mx_, my_ = p2[:, 0], p2[:, 1]
    one = np.ones_like(x)
    zero = np.zeros_like(x)

    for _ in range(num_loops):
        den = A[6] * x + A[7] * y + 1.0
        dx = (A[0] * x + A[1] * y + A[2]) / den - mx_
        dy = (A[3] * x + A[4] * y + A[5]) / den - my_
        w = ((dx * dx + dy * dy) < limit).astype(np.float64)
        Y1 = np.stack([x, y, one, zero, zero, zero, -x * mx_, -y * mx_], -1)
        Y2 = np.stack([zero, zero, zero, x, y, one, -x * my_, -y * my_], -1)
        M = (Y1 * w[:, None]).T @ Y1 + (Y2 * w[:, None]).T @ Y2
        X = (Y1 * (mx_ * w)[:, None]).sum(0) + (Y2 * (my_ * w)[:, None]).sum(0)
        try:
            A = np.linalg.solve(M, X)
        except np.linalg.LinAlgError:
            break

    # geomFuncs.cpp scores its final loop over every point, not just the kept
    # ones, and stores sqrt(err) as match_error
    ax, ay = all1[:, 0], all1[:, 1]
    den = A[6] * ax + A[7] * ay + 1.0
    dx = (A[0] * ax + A[1] * ay + A[2]) / den - all2[:, 0]
    dy = (A[3] * ax + A[4] * ay + A[5]) / den - all2[:, 1]
    err = dx * dx + dy * dy
    m["match_error"] = np.sqrt(err)
    numfit = int((err[keep] < limit).sum())
    return np.append(A, 1.0).reshape(3, 3), numfit
