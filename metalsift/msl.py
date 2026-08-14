"""Custom Metal kernels for every stage of the pipeline.

Ports of cudaSiftD.cu's FindPointsMultiNew, ComputeOrientationsCONST and
ExtractSiftDescriptorsCONSTNew. Three CUDA features have no direct equivalent
and are handled here:

  * texture units      -> hand-written bilinear fetch (`texfetch`) replicating
                          CUDA's cudaFilterModeLinear + cudaAddressModeClamp,
                          which samples around (x-0.5, y-0.5) in pixel space.
  * threadgroup float  -> Metal's atomic_float overloads are device-address-
    atomics             space only, so `tg_add_float` emulates them with a CAS
                          loop on atomic_uint.
  * mutable globals    -> the d_PointCounter chain is replaced by per-octave
                          buffers with their own counters (see sift.py).
"""
import mlx.core as mx

NBLK = 512          # fixed threadgroup count for the grid-stride kernels
MINMAX_W = 30
MINMAX_H = 8

HEADER = """
#include <metal_atomic>
#include <metal_simdgroup>

/* CUDA tex2D<float> with cudaFilterModeLinear + cudaAddressModeClamp and
   unnormalised coords samples around (x-0.5, y-0.5) in pixel-index space.
   Getting this offset wrong drifts keypoints by half a pixel. */
inline float texfetch(const device float *img, int w, int h, float x, float y) {
    float u = x - 0.5f, v = y - 0.5f;
    float fu = metal::floor(u), fv = metal::floor(v);
    float a = u - fu, b = v - fv;
    int x0 = int(fu), y0 = int(fv);
    int x1 = metal::clamp(x0 + 1, 0, w - 1);
    int y1 = metal::clamp(y0 + 1, 0, h - 1);
    x0 = metal::clamp(x0, 0, w - 1);
    y0 = metal::clamp(y0, 0, h - 1);
    float v00 = img[y0 * w + x0], v01 = img[y0 * w + x1];
    float v10 = img[y1 * w + x0], v11 = img[y1 * w + x1];
    return metal::mix(metal::mix(v00, v01, a), metal::mix(v10, v11, a), b);
}

/* threadgroup atomic<float> does not exist in MSL -- CAS on atomic_uint. */
inline void tg_add_float(threadgroup atomic_uint *p, float val) {
    uint old = atomic_load_explicit(p, memory_order_relaxed);
    uint want;
    do {
        want = as_type<uint>(as_type<float>(old) + val);
    } while (!atomic_compare_exchange_weak_explicit(
        p, &old, want, memory_order_relaxed, memory_order_relaxed));
}

inline float tg_load_float(threadgroup atomic_uint *p) {
    return as_type<float>(atomic_load_explicit(p, memory_order_relaxed));
}

/* cudaSiftD.cu:295 -- kept bit-for-bit so descriptor bins match. */
inline float FastAtan2(float y, float x) {
    float absx = metal::abs(x), absy = metal::abs(y);
    float a = metal::min(absx, absy) / metal::max(absx, absy);
    float s = a * a;
    float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
    r = (absy > absx ? 1.57079637f - r : r);
    r = (x < 0.0f ? 3.14159274f - r : r);
    r = (y < 0.0f ? -r : r);
    return r;
}
"""

# ---------------------------------------------------------------------------
# LowPassBlock (cudaSiftD.cu:1986) and ScaleDown (cudaSiftD.cu:84).
#
# Both are separable symmetric filters with clamp-to-edge, and both were
# initially MLX ops. As two compiled shift-multiply-add passes they cost
# 1.30 ms and 1.22 ms at 1920x1080 -- together half the whole extraction --
# because each pass writes a full-resolution intermediate that the next pass
# reads back.
#
# Fusing removes that round trip. Each threadgroup owns a strip of ROWS output
# rows: it does the vertical pass in registers with a sliding window, parks one
# row of partials in threadgroup memory, and finishes horizontally from there.
# Handling 8 rows at once amortises the vertical halo -- a threadgroup reads
# ROWS+8 input rows to produce ROWS outputs, so read amplification is ~2x
# rather than the 9x it would be one row at a time.
#
# CudaSift's LowPassBlock instead does its horizontal pass with warp shuffles
# across a 32-lane row, which costs it 8 of every 32 lanes to the halo. Not
# reproduced: the threadgroup-memory form here is simpler and wastes less.
# ---------------------------------------------------------------------------
LP_W, LP_R, LP_ROWS = 128, 4, 8
SD_OW, SD_R, SD_ROWS = 64, 2, 8

LOWPASS = """
    threadgroup float buff[(128 + 8) * 8];
    const int width = ip[0], height = ip[1];
    const int stride = 128 + 8;

    int tx = int(thread_position_in_threadgroup.x);
    int xp = int(threadgroup_position_in_grid.x) * 128 + tx;
    int y0 = int(threadgroup_position_in_grid.y) * 8;

    if (xp < width + 8) {
        int col = metal::max(metal::min(xp - 4, width - 1), 0);
        float temp[8 + 8];
        for (int i = 0; i < 16; i++)
            temp[i] = img[metal::max(metal::min(y0 + i - 4, height - 1), 0) * width + col];
        for (int r = 0; r < 8; r++) {
            float sum = k[4] * temp[r + 4];
            for (int j = 1; j <= 4; j++)
                sum += k[4 + j] * (temp[r + 4 - j] + temp[r + 4 + j]);
            buff[r * stride + tx] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tx < 128 && xp < width) {
        for (int r = 0; r < 8; r++) {
            int y = y0 + r;
            if (y >= height) break;
            threadgroup float *b = buff + r * stride;
            float sum = k[4] * b[tx + 4];
            for (int j = 1; j <= 4; j++)
                sum += k[4 + j] * (b[tx + 4 - j] + b[tx + 4 + j]);
            out[y * width + xp] = sum;
        }
    }
"""

SCALEDOWN = """
    threadgroup float buff[(2 * 64 + 4) * 8];
    const int width = ip[0], height = ip[1];       /* input dims */
    const int ow = width / 2, oh = height / 2;
    const int stride = 2 * 64 + 4;

    int tx = int(thread_position_in_threadgroup.x);
    int cbase = int(threadgroup_position_in_grid.x) * 128;
    int v0 = int(threadgroup_position_in_grid.y) * 8;

    /* thread tx holds input column cbase + tx - 2 */
    int col = metal::max(metal::min(cbase + tx - 2, width - 1), 0);
    float temp[2 * 8 + 4];
    for (int i = 0; i < 20; i++)
        temp[i] = img[metal::max(metal::min(2 * v0 + i - 2, height - 1), 0) * width + col];
    /* output row v0+r is centred on input row 2*(v0+r) -> temp[2r .. 2r+4] */
    for (int r = 0; r < 8; r++) {
        float sum = 0.0f;
        for (int j = 0; j < 5; j++)
            sum += k[j] * temp[2 * r + j];
        buff[r * stride + tx] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int u = int(threadgroup_position_in_grid.x) * 64 + tx;
    if (tx < 64 && u < ow) {
        for (int r = 0; r < 8; r++) {
            int v = v0 + r;
            if (v >= oh) break;
            threadgroup float *b = buff + r * stride;
            float sum = 0.0f;
            for (int j = 0; j < 5; j++)
                sum += k[j] * b[2 * tx + j];
            out[v * ow + u] = sum;
        }
    }
"""


# ---------------------------------------------------------------------------
# LaplaceMultiMem  (cudaSiftD.cu:1753)
# Expressible as MLX ops, but not efficiently: a depthwise mx.conv2d over the
# 8 scales runs 18.7 ms at 1920x1080, and even a compiled shift-multiply-add
# costs 7.9 ms because the 8 blurred planes (66 MB) get materialised and
# re-read. Fusing as CudaSift does keeps the vertical pass in registers and
# writes only the 7 DoG planes.
#
# CUDA's kern[LAPLACE_S][LAPLACE_R+1] register array is not reproduced here:
# it only works because both loops fully unroll, and a thread-space array that
# fails to unroll spills to thread-local memory. The taps are re-read from the
# input array instead, where every thread hits the same address and broadcasts.
# ---------------------------------------------------------------------------
LAPLACE_W = 128
LAPLACE_R = 4
LAPLACE_S = 8

LAPLACE = """
    threadgroup float buff[(128 + 8) * 8];

    const int width = ip[0], height = ip[1];
    int tx = int(thread_position_in_threadgroup.x);
    int xp = int(threadgroup_position_in_grid.x) * 128 + tx;
    int yp = int(threadgroup_position_in_grid.y);
    int pitch = width;

    if (xp < width + 8) {
        int col = metal::max(metal::min(xp - 4, width - 1), 0);
        float temp[9];
        for (int i = 0; i <= 8; i++)
            temp[i] = img[metal::max(metal::min(yp + i - 4, height - 1), 0) * pitch + col];
        for (int s = 0; s < 8; s++) {
            /* taps are stored as the full symmetric 9, centre at index 4 */
            float sum = k[s * 9 + 4] * temp[4];
            for (int j = 1; j <= 4; j++)
                sum += k[s * 9 + 4 + j] * (temp[4 - j] + temp[4 + j]);
            buff[136 * s + tx] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tx < 128 && xp < width) {
        float oldRes = k[4] * buff[tx + 4];
        for (int j = 1; j <= 4; j++)
            oldRes += k[4 + j] * (buff[tx + 4 - j] + buff[tx + 4 + j]);
        for (int s = 1; s < 8; s++) {
            threadgroup float *b = buff + 136 * s;
            float res = k[s * 9 + 4] * b[tx + 4];
            for (int j = 1; j <= 4; j++)
                res += k[s * 9 + 4 + j] * (b[tx + 4 - j] + b[tx + 4 + j]);
            dog[(s - 1) * height * pitch + yp * pitch + xp] = res - oldRes;
            oldRes = res;
        }
    }
"""


# ---------------------------------------------------------------------------
# FindPointsMultiNew  (cudaSiftD.cu:1292)
# threadgroup = 32 threads = exactly one simdgroup, as in CUDA (MINMAX_W + 2).
# ---------------------------------------------------------------------------
FIND_POINTS = """
    threadgroup ushort pts[64];

    const int width = ip[0], height = ip[1], maxPts = ip[2];
    const float subsampling = fp[0], lowestScale = fp[1], thresh = fp[2];
    const float factor = fp[3], edgeLimit = fp[4];

    int tx  = int(thread_position_in_threadgroup.x);
    int bxg = int(threadgroup_position_in_grid.x);
    int byg = int(threadgroup_position_in_grid.y);
    int block = bxg / 5;
    int scale = bxg - 5 * block;
    int minx = block * 30;
    int maxx = metal::min(minx + 30, width);
    int xpos = minx + tx;
    int pitch = width;
    int size = pitch * height;
    int base = size * scale + metal::max(metal::min(xpos - 1, width - 1), 0);
    int yloops = metal::min(height - 8 * byg, 8);

    /* cheap simdgroup-wide reject before touching the 3x3x3 neighbourhood */
    float maxv = 0.0f;
    for (int y = 0; y < yloops; y++) {
        int ypos = 8 * byg + y;
        maxv = metal::max(maxv, metal::abs(dog[base + ypos * pitch + size]));
    }
    if (!simd_any(maxv > thresh)) return;

    int ptbits = 0;
    for (int y = 0; y < yloops; y++) {
        int ypos = 8 * byg + y;
        int yptr1 = base + ypos * pitch;
        float d11 = dog[yptr1 + size];
        if (simd_any(metal::abs(d11) > thresh)) {
            int yptr0 = base + metal::max(0, ypos - 1) * pitch;
            int yptr2 = base + metal::min(height - 1, ypos + 1) * pitch;
            float d01 = dog[yptr1];
            float d10 = dog[yptr0 + size];
            float d12 = dog[yptr2 + size];
            float d21 = dog[yptr1 + 2 * size];
            float d00 = dog[yptr0];
            float d02 = dog[yptr2];
            float ymin1 = metal::min(metal::min(d00, d01), d02);
            float ymax1 = metal::max(metal::max(d00, d01), d02);
            float d20 = dog[yptr0 + 2 * size];
            float d22 = dog[yptr2 + 2 * size];
            float ymin3 = metal::min(metal::min(d20, d21), d22);
            float ymax3 = metal::max(metal::max(d20, d21), d22);
            float ymin2 = metal::min(metal::min(ymin1,
                          metal::min(metal::min(d10, d12), d11)), ymin3);
            float ymax2 = metal::max(metal::max(ymax1,
                          metal::max(metal::max(d10, d12), d11)), ymax3);
            /* lanes 0 and 31 read out of range here; their results are
               discarded by the tx guard below, exactly as in CUDA. */
            float nmin2 = metal::min(simd_shuffle_up(ymin2, 1),
                                     simd_shuffle_down(ymin2, 1));
            float nmax2 = metal::max(simd_shuffle_up(ymax2, 1),
                                     simd_shuffle_down(ymax2, 1));
            float minv = metal::min(metal::min(nmin2, ymin1), ymin3);
            minv = metal::min(metal::min(minv, d10), d12);
            float mxv = metal::max(metal::max(nmax2, ymax1), ymax3);
            mxv = metal::max(metal::max(mxv, d10), d12);
            if (tx > 0 && tx < 31 && xpos <= maxx)
                ptbits |= int((d11 < metal::min(-thresh, minv)) ||
                              (d11 > metal::max(thresh, mxv))) << y;
        }
    }

    /* popcount + prefix sum compaction, verbatim from the CUDA idiom */
    uint numbits = uint(metal::popcount(ptbits));
    uint incl = simd_prefix_inclusive_sum(numbits);
    uint pos = incl - numbits;
    for (int y = 0; y < yloops; y++) {
        int ypos = 8 * byg + y;
        if ((ptbits & (1 << y)) != 0 && pos < 32u) {
            pts[2 * pos + 0] = ushort(xpos - 1);
            pts[2 * pos + 1] = ushort(ypos);
            pos++;
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint total = simd_shuffle(incl, 31);

    if (uint(tx) >= total) return;
    int px = int(pts[2 * tx + 0]);
    int py = int(pts[2 * tx + 1]);
    /* CUDA reads 1px outside the image here and lands in pitch padding;
       MLX arrays are tight, so reject border candidates instead. */
    if (px < 1 || px >= width - 1 || py < 1 || py >= height - 1) return;

    int p = px + (py + (scale + 1) * height) * pitch;
    float val = dog[p];
    float dxx = 2.0f * val - dog[p - 1] - dog[p + 1];
    float dyy = 2.0f * val - dog[p - pitch] - dog[p + pitch];
    float dxy = 0.25f * (dog[p + pitch + 1] + dog[p - pitch - 1]
                       - dog[p - pitch + 1] - dog[p + pitch - 1]);
    float tra = dxx + dyy;
    float det = dxx * dyy - dxy * dxy;
    if (!(tra * tra < edgeLimit * det)) return;

    float edge = tra * tra / det;
    float dx = 0.5f * (dog[p + 1] - dog[p - 1]);
    float dy = 0.5f * (dog[p + pitch] - dog[p - pitch]);
    int p0 = p - height * pitch;
    int p2 = p + height * pitch;
    float ds = 0.5f * (dog[p0] - dog[p2]);
    float dss = 2.0f * val - dog[p2] - dog[p0];
    float dxs = 0.25f * (dog[p2 + 1] + dog[p0 - 1] - dog[p0 + 1] - dog[p2 - 1]);
    float dys = 0.25f * (dog[p2 + pitch] + dog[p0 - pitch]
                       - dog[p2 - pitch] - dog[p0 + pitch]);
    float idxx = dyy * dss - dys * dys;
    float idxy = dys * dxs - dxy * dss;
    float idxs = dxy * dys - dyy * dxs;
    float idet = 1.0f / (idxx * dxx + idxy * dxy + idxs * dxs);
    float idyy = dxx * dss - dxs * dxs;
    float idys = dxy * dxs - dxx * dys;
    float idss = dxx * dyy - dxy * dxy;
    float pdx = idet * (idxx * dx + idxy * dy + idxs * ds);
    float pdy = idet * (idxy * dx + idyy * dy + idys * ds);
    float pds = idet * (idxs * dx + idys * dy + idss * ds);
    if (pdx < -0.5f || pdx > 0.5f || pdy < -0.5f ||
        pdy > 0.5f || pds < -0.5f || pds > 0.5f) {
        pdx = dx / dxx;
        pdy = dy / dyy;
        pds = ds / dss;
    }
    float dval = 0.5f * (dx * pdx + dy * pdy + ds * pds);
    float sc = metal::exp2(float(scale) / 5.0f) * metal::exp2(pds * factor);
    if (sc < lowestScale) return;

    uint idx = atomic_fetch_add_explicit(&cnt[0], 1u, memory_order_relaxed);
    if (idx >= uint(maxPts)) return;
    atomic_store_explicit(&kp[idx * 6 + 0], float(px) + pdx, memory_order_relaxed);
    atomic_store_explicit(&kp[idx * 6 + 1], float(py) + pdy, memory_order_relaxed);
    atomic_store_explicit(&kp[idx * 6 + 2], sc, memory_order_relaxed);
    atomic_store_explicit(&kp[idx * 6 + 3], val + dval, memory_order_relaxed);
    atomic_store_explicit(&kp[idx * 6 + 4], edge, memory_order_relaxed);
    atomic_store_explicit(&kp[idx * 6 + 5], subsampling, memory_order_relaxed);
"""

# ---------------------------------------------------------------------------
# ComputeOrientationsCONST  (cudaSiftD.cu:972)
# 128 threads instead of CUDA's 121: the yd<11 guard already excludes the
# extra 7, so the result is identical and the threadgroup stays simd-aligned.
#
# CUDA appends the extra keypoints for secondary orientation peaks into the
# very buffer it is reading from, while other threadgroups are still reading
# it. That hazard does not exist here: `kp` is a const input and `kp_out` is a
# distinct output buffer, so this kernel can read one and append to the other
# freely. Within kp_out, the copy region [0, n0) and the append region
# [n0, maxPts) are disjoint by construction, since the append index is always
# n0 + atomic_fetch_add.
# ---------------------------------------------------------------------------
ORIENTATIONS = """
    threadgroup atomic_uint hist_a[32];
    threadgroup float hist[64];
    threadgroup float gauss[11];

    const int width = ip[0], height = ip[1], maxPts = ip[2];
    int tx = int(thread_position_in_threadgroup.x);
    uint totPts = metal::min(cnt[0], uint(maxPts));
    uint nb = threadgroups_per_grid.x;

    for (uint bx = threadgroup_position_in_grid.x; bx < totPts; bx += nb) {
        float kscale = kp[bx * 6 + 2];
        float i2sigma2 = -1.0f / (2.0f * 1.5f * 1.5f * kscale * kscale);
        if (tx < 11)
            gauss[tx] = metal::exp(i2sigma2 * float((tx - 5) * (tx - 5)));
        if (tx < 32)
            atomic_store_explicit(&hist_a[tx], as_type<uint>(0.0f), memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* NB: no +0.5 here, unlike the descriptor kernel -- CudaSift's
           orientation window really is centred half a pixel off. Faithful. */
        float xp = kp[bx * 6 + 0] - 4.5f;
        float yp = kp[bx * 6 + 1] - 4.5f;
        int yd = tx / 11;
        int xd = tx - yd * 11;
        if (yd < 11) {
            float xf = xp + float(xd);
            float yf = yp + float(yd);
            float dx = texfetch(img, width, height, xf + 1.0f, yf)
                     - texfetch(img, width, height, xf - 1.0f, yf);
            float dy = texfetch(img, width, height, xf, yf + 1.0f)
                     - texfetch(img, width, height, xf, yf - 1.0f);
            int bin = int(16.0f * metal::atan2(dy, dx) / 3.1416f + 16.5f);
            if (bin > 31) bin = 0;
            float grad = metal::sqrt(dx * dx + dy * dy);
            tg_add_float(&hist_a[bin], grad * gauss[xd] * gauss[yd]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tx < 32) hist[tx] = tg_load_float(&hist_a[tx]);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        int x1m = (tx >= 1 ? tx - 1 : tx + 31);
        int x1p = (tx <= 30 ? tx + 1 : tx - 31);
        if (tx < 32) {
            int x2m = (tx >= 2 ? tx - 2 : tx + 30);
            int x2p = (tx <= 29 ? tx + 2 : tx - 30);
            hist[tx + 32] = 6.0f * hist[tx] + 4.0f * (hist[x1m] + hist[x1p])
                          + (hist[x2m] + hist[x2p]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tx < 32) {
            float v = hist[32 + tx];
            hist[tx] = (v > hist[32 + x1m] && v >= hist[32 + x1p] ? v : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tx == 0) {
            float maxval1 = 0.0f, maxval2 = 0.0f;
            int i1 = -1, i2 = -1;
            for (int i = 0; i < 32; i++) {
                float v = hist[i];
                if (v > maxval1) {
                    maxval2 = maxval1; maxval1 = v;
                    i2 = i1; i1 = i;
                } else if (v > maxval2) {
                    maxval2 = v; i2 = i;
                }
            }
            float a1 = hist[32 + ((i1 + 1) & 31)];
            float a2 = hist[32 + ((i1 + 31) & 31)];
            float peak = float(i1) + 0.5f * (a1 - a2) / (2.0f * maxval1 - a1 - a2);

            for (int c = 0; c < 6; c++)
                atomic_store_explicit(&kp_out[bx * 7 + c], kp[bx * 6 + c],
                                      memory_order_relaxed);
            atomic_store_explicit(&kp_out[bx * 7 + 6],
                                  11.25f * (peak < 0.0f ? peak + 32.0f : peak),
                                  memory_order_relaxed);

            if (maxval2 > 0.8f * maxval1) {
                float b1 = hist[32 + ((i2 + 1) & 31)];
                float b2 = hist[32 + ((i2 + 31) & 31)];
                float pk2 = float(i2) + 0.5f * (b1 - b2) / (2.0f * maxval2 - b1 - b2);
                uint idx = totPts + atomic_fetch_add_explicit(&dup[0], 1u,
                                                              memory_order_relaxed);
                if (idx < uint(maxPts)) {
                    for (int c = 0; c < 6; c++)
                        atomic_store_explicit(&kp_out[idx * 7 + c], kp[bx * 6 + c],
                                              memory_order_relaxed);
                    atomic_store_explicit(&kp_out[idx * 7 + 6],
                                          11.25f * (pk2 < 0.0f ? pk2 + 32.0f : pk2),
                                          memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""

# ---------------------------------------------------------------------------
# ExtractSiftDescriptorsCONSTNew  (cudaSiftD.cu:308)
# threadgroup (16, 8) = 128 threads = 4 simdgroups, as in CUDA.
# ---------------------------------------------------------------------------
DESCRIPTORS = """
    threadgroup float gauss[16];
    threadgroup atomic_uint buf[128];
    threadgroup float sums[4];

    const int width = ip[0], height = ip[1], maxPts = ip[2];
    int tx = int(thread_position_in_threadgroup.x);
    int ty = int(thread_position_in_threadgroup.y);
    int idx = ty * 16 + tx;
    if (ty == 0)
        gauss[tx] = metal::fast::exp(-(float(tx) - 7.5f) * (float(tx) - 7.5f) / 128.0f);

    uint totPts = metal::min(cnt[0], uint(maxPts));
    uint nb = threadgroups_per_grid.x;

    for (uint bx = threadgroup_position_in_grid.x; bx < totPts; bx += nb) {
        atomic_store_explicit(&buf[idx], as_type<uint>(0.0f), memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float theta = 2.0f * 3.1415f / 360.0f * kp[bx * 7 + 6];
        float sina = metal::fast::sin(theta);
        float cosa = metal::fast::cos(theta);
        float sc = 12.0f / 16.0f * kp[bx * 7 + 2];
        float ssina = sc * sina, scosa = sc * cosa;
        float kx = kp[bx * 7 + 0], ky = kp[bx * 7 + 1];

        for (int y = ty; y < 16; y += 8) {
            /* the +0.5 the orientation kernel omits -- see cudaSiftD.cu:338 */
            float xpos = kx + (float(tx) - 7.5f) * scosa - (float(y) - 7.5f) * ssina + 0.5f;
            float ypos = ky + (float(tx) - 7.5f) * ssina + (float(y) - 7.5f) * scosa + 0.5f;
            float dx = texfetch(img, width, height, xpos + cosa, ypos + sina)
                     - texfetch(img, width, height, xpos - cosa, ypos - sina);
            float dy = texfetch(img, width, height, xpos - sina, ypos + cosa)
                     - texfetch(img, width, height, xpos + sina, ypos - cosa);
            float grad = gauss[y] * gauss[tx] * metal::sqrt(dx * dx + dy * dy);
            float angf = 4.0f / 3.1415f * FastAtan2(dy, dx) + 4.0f;

            int hori = (tx + 2) / 4 - 1;
            float horf = (float(tx) - 1.5f) / 4.0f - float(hori);
            float ihorf = 1.0f - horf;
            int veri = (y + 2) / 4 - 1;
            float verf = (float(y) - 1.5f) / 4.0f - float(veri);
            float iverf = 1.0f - verf;
            /* CudaSift scales by 4/3.1415 while FastAtan2 reflects about
               3.14159274, so angf actually spans [-0.000118, 8.000118], not
               [0, 8]. With C truncation that yields two defects at gradients
               within ~3e-5 rad of +/-pi:
                 angf = -1.18e-4 -> angi = 0, so angf stays negative and a
                   negative weight is accumulated into the histogram, which
                   surfaces as a descriptor bin around -1e-6;
                 angf = 8.000118 -> angi = 8, so p1 = angi + hbin reaches 128
                   and writes past buf[128].
               floor plus a mask fixes both and is what the circular histogram
               wanted anyway: bin 8 is bin 0, and bin -1 is bin 7. The upstream
               constant is left alone -- it biases bin assignment by 2.4e-4 of
               a bin, which is irrelevant, and changing it would perturb every
               descriptor rather than just the broken ones. */
            int angi = int(metal::floor(angf));
            angf -= float(angi);              /* now strictly in [0, 1) */
            angi &= 7;                        /* wrap 8 -> 0, -1 -> 7 */
            int angp = (angi + 1) & 7;
            float iangf = 1.0f - angf;

            int hbin = 8 * (4 * veri + hori);
            int p1 = angi + hbin;
            int p2 = angp + hbin;
            if (tx >= 2) {
                float grad1 = ihorf * grad;
                if (y >= 2) {
                    float grad2 = iverf * grad1;
                    tg_add_float(&buf[p1], iangf * grad2);
                    tg_add_float(&buf[p2], angf * grad2);
                }
                if (y <= 13) {
                    float grad2 = verf * grad1;
                    tg_add_float(&buf[p1 + 32], iangf * grad2);
                    tg_add_float(&buf[p2 + 32], angf * grad2);
                }
            }
            if (tx <= 13) {
                float grad1 = horf * grad;
                if (y >= 2) {
                    float grad2 = iverf * grad1;
                    tg_add_float(&buf[p1 + 8], iangf * grad2);
                    tg_add_float(&buf[p2 + 8], angf * grad2);
                }
                if (y <= 13) {
                    float grad2 = verf * grad1;
                    tg_add_float(&buf[p1 + 40], iangf * grad2);
                    tg_add_float(&buf[p2 + 40], angf * grad2);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        /* normalise twice, clipping at 0.2 in between (Lowe's recipe) */
        float bval = tg_load_float(&buf[idx]);
        float s = simd_sum(bval * bval);
        if ((idx & 31) == 0) sums[idx / 32] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
        tsum1 = metal::min(bval * metal::rsqrt(tsum1), 0.2f);

        /* sums[] is about to be reused for the second reduction, and without
           this barrier a simdgroup that has finished reading it can overwrite
           sums[0] while another is still summing it for tsum1 -- giving that
           simdgroup a corrupted normalisation factor. It made the kernel
           nondeterministic to ~0.2 per bin on identical input. CudaSift has
           the same missing sync at cudaSiftD.cu:397-404. */
        threadgroup_barrier(mem_flags::mem_threadgroup);

        s = simd_sum(tsum1 * tsum1);
        if ((idx & 31) == 0) sums[idx / 32] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
        desc[bx * 128 + idx] = tsum1 * metal::rsqrt(tsum2);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""


def _k(name, inputs, outputs, source, atomic=False):
    return mx.fast.metal_kernel(
        name=name, input_names=inputs, output_names=outputs,
        header=HEADER, source=source, atomic_outputs=atomic,
    )


_lowpass = _k("lowpass", ["img", "k", "ip"], ["out"], LOWPASS)
_scaledown = _k("scale_down", ["img", "k", "ip"], ["out"], SCALEDOWN)
_laplace = _k("laplace_multi", ["img", "k", "ip"], ["dog"], LAPLACE)
_find = _k("find_points", ["dog", "ip", "fp"], ["kp", "cnt"], FIND_POINTS, atomic=True)
_orient = _k("compute_orientations", ["img", "kp", "cnt", "ip"],
             ["kp_out", "dup"], ORIENTATIONS, atomic=True)
_desc = _k("descriptors", ["img", "kp", "cnt", "ip"], ["desc"], DESCRIPTORS)


def _idiv(a, b):
    return (a + b - 1) // b


def lowpass(img, taps):
    """taps: (9,) float32 MLX array. Returns (h, w), same shape as img."""
    h, w = img.shape
    ip = mx.array([w, h], dtype=mx.int32)
    (out,) = _lowpass(
        inputs=[img, taps, ip],
        grid=((LP_W + 2 * LP_R) * _idiv(w, LP_W), _idiv(h, LP_ROWS), 1),
        threadgroup=(LP_W + 2 * LP_R, 1, 1),
        output_shapes=[(h, w)], output_dtypes=[mx.float32], init_value=0.0,
    )
    return out


def scale_down(img, taps):
    """taps: (5,) float32 MLX array. Returns (h//2, w//2)."""
    h, w = img.shape
    ip = mx.array([w, h], dtype=mx.int32)
    (out,) = _scaledown(
        inputs=[img, taps, ip],
        grid=((2 * SD_OW + 2 * SD_R) * _idiv(w // 2, SD_OW),
              _idiv(h // 2, SD_ROWS), 1),
        threadgroup=(2 * SD_OW + 2 * SD_R, 1, 1),
        output_shapes=[(h // 2, w // 2)], output_dtypes=[mx.float32],
        init_value=0.0,
    )
    return out


def laplace_multi(img, taps, w, h):
    """taps: (8, 9) float32 MLX array. Returns (7, h, w) DoG planes."""
    ip = mx.array([w, h], dtype=mx.int32)
    nbx = _idiv(w, LAPLACE_W)
    (dog,) = _laplace(
        inputs=[img, taps, ip],
        grid=((LAPLACE_W + 2 * LAPLACE_R) * nbx, h, 1),
        threadgroup=(LAPLACE_W + 2 * LAPLACE_R, 1, 1),
        output_shapes=[(LAPLACE_S - 1, h, w)],
        output_dtypes=[mx.float32],
        init_value=0.0,
    )
    return dog


def find_points(dog, w, h, max_pts, subsampling, lowest_scale, thresh,
                factor, edge_limit):
    ip = mx.array([w, h, max_pts], dtype=mx.int32)
    fp = mx.array([subsampling, lowest_scale, thresh, factor, edge_limit],
                  dtype=mx.float32)
    nbx = _idiv(w, MINMAX_W) * 5
    nby = _idiv(h, MINMAX_H)
    return _find(
        inputs=[dog, ip, fp],
        grid=(32 * nbx, nby, 1), threadgroup=(32, 1, 1),
        output_shapes=[(max_pts, 6), (1,)],
        output_dtypes=[mx.float32, mx.uint32],
        init_value=0,
    )


def compute_orientations(img, kp, cnt, w, h, max_pts):
    """Assign orientations and emit the (maxPts, 7) keypoint array.

    Also appends a keypoint for every secondary orientation peak, which
    CudaSift does in a second pass over the shared buffer. Returns
    (kp_out, total) with total on device, so no sync is needed mid-pipeline.
    """
    ip = mx.array([w, h, max_pts], dtype=mx.int32)
    kp_out, dup = _orient(
        inputs=[img, kp, cnt, ip],
        grid=(128 * NBLK, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(max_pts, 7), (1,)],
        output_dtypes=[mx.float32, mx.uint32],
        init_value=0,
    )
    cap = mx.array([max_pts], dtype=mx.uint32)
    return kp_out, mx.minimum(mx.minimum(cnt, cap) + dup, cap)


def descriptors(img, kp, cnt, w, h, max_pts):
    ip = mx.array([w, h, max_pts], dtype=mx.int32)
    (desc,) = _desc(
        inputs=[img, kp, cnt, ip],
        grid=(16 * NBLK, 8, 1), threadgroup=(16, 8, 1),
        output_shapes=[(max_pts, 128)],
        output_dtypes=[mx.float32],
        init_value=0.0,
    )
    return desc
