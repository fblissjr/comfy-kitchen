/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Sol-Attn preprocess: post-rope q/k/v -> the workspace carriers
//   qiP  [B,T,H,D] int8   (perm_d)            qs   [B,T,H] f32
//   kiP  [B*H,Tp,D] int8  (perm_key + perm_d) ksb  [B*H,Tp] float2 = (ks, bias)
//   vTi  [B*H,D,Tp] int8  (perm_d on keys)    vsc  [B*H,D] f32  -- sol_attn_vtranspose.cu
//   kciP [B*H,NPAD,D] int8 (perm_d)           kcs  [B*H,NPAD] f32
//   vcT  [B*H,D,NPAD] bf16                    threshold [B*H,NQ] f32
//   cen8 [B*H,NQ,D] int8  (perm_d)            cens [B*H,NQ] f32
// Q-side carriers are indexed by T, K/V-side by Tp. Inputs are read through
// explicit strides (last dim contiguous). K's mean and V's scale are global
// reductions, hence the separate passes. `launch_sol_finish` is the chunked
// producer's second half (pooled sums -> means, pooled quant, threshold).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include "sol_layout.cuh"

namespace {
using namespace sol;

constexpr int HD = HEAD_DIM, BLK = BLOCK;

// Scratch: pooled K block means (sums in the producer path), kmean, kcvar;
// then the q/k balance's per-block sums of squares (K's, then reused for
// Q's) and its two factors, written only when balancing is on.
struct Scratch { float* kc; float* kmean; float* kcvar; float* sq; float* fq; float* fk; };
inline Scratch carve_scratch(void* scratch, int B, int H, int NPAD) {
    Scratch s;
    s.kc = (float*)scratch;
    s.kmean = s.kc + (size_t)B * H * NPAD * HD;
    s.kcvar = s.kmean + (size_t)B * H * HD;
    s.sq = s.kcvar + (size_t)B * H * HD;
    s.fq = s.sq + (size_t)B * H * NPAD * HD;
    s.fk = s.fq + (size_t)B * H * HD;
    return s;
}

// routing threshold: tau sigma of the proxy row, from the centroid's proxy variance
__device__ __forceinline__ float thr_of(float var, float tau, float log2s) {
    return tau * sqrtf(var * log2s * log2s + 1e-6f);
}

// ---- pass 1: K/V block reductions + V's per-channel |max| ----
// One block per (bh, pooled block), thread == channel.
template <typename E>
__global__ void prep_reduce_kv(const E* __restrict__ k,
                               const E* __restrict__ v,
                               float* __restrict__ kc, __nv_bfloat16* __restrict__ vcT,
                               float* __restrict__ vamax, float* __restrict__ ksq,   // ksq: block sum of K^2, or null
                               const int32_t* __restrict__ blen,
                               int T, int H, int NTB, int NPAD,
                               int64_t sb, int64_t st, int64_t sh,
                               int64_t vb, int64_t vt, int64_t vh) {
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H, d = threadIdx.x;
    const size_t o = ((size_t)bh * NPAD + n) * HD + d;
    if (n >= NTB) {
        kc[o] = 0.f;
        if (ksq) ksq[o] = 0.f;
        vcT[((size_t)bh * HD + d) * NPAD + n] = __float2bfloat16(0.f);
        return;
    }

    const int t0 = n * BLK, len = block_len_of(blen, n, T);
    float sk = 0.f, sk2 = 0.f, sv = 0.f, av = 0.f;
    for (int i = 0; i < len; ++i) {
        const int64_t voff = batch * vb + (int64_t)(t0 + i) * vt + head * vh + d;
        const float kk = to_f32(k[batch * sb + (int64_t)(t0 + i) * st + head * sh + d]);
        const float vv = to_f32(v[voff]);
        sk += kk; sk2 = fmaf(kk, kk, sk2); sv += vv; av = fmaxf(av, fabsf(vv));
    }
    kc[o] = sk / (float)len;    // block MEAN of K
    if (ksq) ksq[o] = sk2;
    vcT[((size_t)bh * HD + d) * NPAD + n] = __float2bfloat16(sv);   // block SUM of V
    atomicMax(reinterpret_cast<unsigned int*>(&vamax[(size_t)bh * HD + d]),
              __float_as_uint(av));   // av >= 0, so the bit pattern orders correctly
}

// ---- pass 2: reductions over the pooled tensors ----
// vamax (pass 1's |max|; null in the producer path, whose caller supplies the
// V scale) becomes the per-channel V scale; may alias vsc.
__global__ void prep_pooled_stats(const float* __restrict__ kc,
                                  float* __restrict__ kmean,
                                  const float* vamax, float* vsc,
                                  float* __restrict__ kcvar, int NTB, int NPAD) {
    const int bh = blockIdx.x, d = threadIdx.x;
    float sm = 0.f, ss = 0.f;
    for (int n = 0; n < NTB; ++n) {
        const float x = kc[((size_t)bh * NPAD + n) * HD + d];
        sm += x;
        ss = fmaf(x, x, ss);
    }
    const float m = sm / (float)NTB;
    kmean[(size_t)bh * HD + d] = m;
    if (vamax) vsc[(size_t)bh * HD + d] = fmaxf(vamax[(size_t)bh * HD + d] / 127.0f, 1e-8f);
    kcvar[(size_t)bh * HD + d] = fmaxf(ss / (float)NTB - m * m, 0.f);
}

// ---- pass 3: centre + quantize the pooled keys ----
__global__ void prep_pooled_quant(const float* __restrict__ kc,
                                  const float* __restrict__ kmean,
                                  const float* __restrict__ fk,   // [B*H, HD] balance, or null
                                  int8_t* __restrict__ kciP, float* __restrict__ kcs,
                                  int NTB, int NPAD) {
    __shared__ float sred[HD];
    const int n = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const size_t o = ((size_t)bh * NPAD + n) * HD + d;
    const bool live = n < NTB;
    const float f = fk ? fk[(size_t)bh * HD + d] : 1.f;
    const float x = live ? (kc[o] - kmean[(size_t)bh * HD + d]) * f : 0.f;
    const float sc = fmaxf(block_max128(fabsf(x), sred) / 127.0f, 1e-12f);
    if (d == 0) kcs[(size_t)bh * NPAD + n] = live ? sc : 0.f;
    kciP[((size_t)bh * NPAD + n) * HD + perm_d(d)] = live ? q8(x, 1.f / sc) : (int8_t)0;
}

// ---- balance (qk_balance only): per-(bh, channel) factors from the sequence ----
// Q's per-block sums of squares, same shape as pass 1's K sums so one
// reduction serves both. Grid (NQ, B*H), thread == channel.
template <typename E>
__global__ void prep_reduce_q_sq(const E* __restrict__ q, float* __restrict__ part,
                                 const int32_t* __restrict__ blen,
                                 int T, int H, int NPAD,
                                 int64_t sb, int64_t st, int64_t sh) {
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H, d = threadIdx.x;
    const int t0 = n * BLK, len = block_len_of(blen, n, T);
    float s = 0.f;
    for (int i = 0; i < len; ++i) {
        const float x = to_f32(q[batch * sb + (int64_t)(t0 + i) * st + head * sh + d]);
        s = fmaf(x, x, s);
    }
    part[((size_t)bh * NPAD + n) * HD + d] = s;
}
// Sum the per-block partials over the N live blocks, one CTA per (bh).
// Fixed order, so the factor -- and every int8 byte behind it -- is the
// same on every call with the same inputs.
__global__ void prep_balance_reduce(const float* __restrict__ part, float* __restrict__ out,
                                    int N, int NPAD) {
    const int bh = blockIdx.x, d = threadIdx.x;
    float s = 0.f;
    for (int n = 0; n < N; ++n) s += part[((size_t)bh * NPAD + n) * HD + d];
    out[(size_t)bh * HD + d] = s;
}
// The factor itself, from the two sums of squares (K's in `ksq2`, Q's in
// `qsq2`, both [B*H, HD]); `fq` and `fk` may alias either input, each
// thread reads its channel before it writes. The gate, the normalization
// and the clamps mirror the reference (backends/eager/sol_attn.py,
// qk_balance_factor); the summation order differs, so the two agree to fp32
// rounding, not bit for bit.
__global__ void prep_balance_factor(const float* ksq2, const float* qsq2,
                                    float* fq, float* fk) {
    __shared__ float sred[HD];
    __shared__ float se[HD];
    const int bh = blockIdx.x, d = threadIdx.x;
    const float sk2 = ksq2[(size_t)bh * HD + d];
    const float sq2 = qsq2[(size_t)bh * HD + d];
    // gate: energy share of the BAL_TOP loudest K channels (ties by index)
    se[d] = sk2;
    __syncthreads();
    int rank = 0;
    for (int j = 0; j < HD; ++j) rank += (se[j] > sk2) || (se[j] == sk2 && j < d);
    const float top = block_sum128(rank < BAL_TOP ? sk2 : 0.f, sred);
    const float tot = block_sum128(sk2, sred);
    const bool open = tot > 0.f && top >= BAL_MIN_SHARE * tot;
    const float rk = fmaxf(sqrtf(sk2), 1e-6f), rq = fmaxf(sqrtf(sq2), 1e-6f);
    float f = powf(rk, BAL_ALPHA) / powf(rq, 1.f - BAL_ALPHA);
    const float lmean = block_sum128(logf(f), sred) / (float)HD;
    f = open ? f / expf(lmean) : 1.f;
    fq[(size_t)bh * HD + d] = f;
    fk[(size_t)bh * HD + d] = 1.f / f;
}

// ---- pass 4: quantize Q, centroid, routing threshold from one staged tile ----
template <typename E>
__global__ void prep_q(const E* __restrict__ q, const float* __restrict__ kcvar,
                       int8_t* __restrict__ qiP, float* __restrict__ qs,
                       float* __restrict__ thr,
                       int8_t* __restrict__ cen8, float* __restrict__ cens,
                       float* __restrict__ qmean,   // [B*H, NPAD, HD] f32 block means
                       const float* __restrict__ fq,   // [B*H, HD] balance, or null
                       const int32_t* __restrict__ blen,
                       int T, int H, int NQ, int NPAD, float tau, float log2s,
                       int64_t sb, int64_t st, int64_t sh) {
    __shared__ __align__(16) E sQ[BLK * LD_TILE];
    __shared__ __align__(16) float sred[HD];
    __shared__ float sf[HD];
    const int qb = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int t0 = qb * BLK, len = block_len_of(blen, qb, T), nrows = min(BLK, T - t0);

    stage_tile64(sQ, q + batch * sb + (int64_t)t0 * st + head * sh, st, len);
    sf[threadIdx.x] = fq ? fq[(size_t)bh * HD + threadIdx.x] : 1.f;
    __syncthreads();
    const size_t tok0 = (size_t)batch * T + t0;
    quant_q_rows(sQ, len, nrows, qiP + (tok0 * H + head) * HD, qs + tok0 * H + head, H,
                 fq ? sf : nullptr);
    __syncthreads();
    const size_t qrow = (size_t)bh * NQ + qb;
    const float c = centroid_quant(sQ, len, sred, cen8 + qrow * HD, cens + qrow, sf[threadIdx.x]);
    qmean[((size_t)bh * NPAD + qb) * HD + threadIdx.x] = c;
    __syncthreads();                       // sred held the centroid bytes
    const float var = block_sum128(c * c * kcvar[(size_t)bh * HD + threadIdx.x], sred);
    if (threadIdx.x == 0) thr[qrow] = thr_of(var, tau, log2s);
}

// ---- pass 5: centre + quantize K into the permuted layout ----
template <typename E>
__global__ void prep_k(const E* __restrict__ k, const float* __restrict__ kmean,
                       int8_t* __restrict__ kiP, float2* __restrict__ ksb,
                       const float* __restrict__ kbias,   // [B, T] log2 units, or null
                       const float* __restrict__ fk,      // [B*H, HD] balance, or null
                       const int32_t* __restrict__ blen,
                       int T, int Tp, int H,
                       int64_t sb, int64_t st, int64_t sh) {
    __shared__ __align__(16) E sK[BLK * LD_TILE];
    __shared__ float sf[HD];
    const int n = blockIdx.x, bh = blockIdx.y;
    const int batch = bh / H, head = bh % H;
    const int t0 = n * BLK, len = block_len_of(blen, n, T);

    stage_tile64(sK, k + batch * sb + (int64_t)t0 * st + head * sh, st, len);
    sf[threadIdx.x] = fk ? fk[(size_t)bh * HD + threadIdx.x] : 1.f;
    __syncthreads();
    const size_t dst0 = (size_t)bh * Tp + n * BLK;
    quant_k_rows(sK, len, kmean + (size_t)bh * HD,
                 kbias ? kbias + (size_t)batch * T + t0 : nullptr,
                 kiP + dst0 * HD, ksb + dst0, fk ? sf : nullptr);
}

// ---- producer-path finish: pooled sums -> means, next-step kmean ----
__global__ void prep_sums_to_means(float* __restrict__ kc,
                                   float* __restrict__ kmean_next,
                                   const int32_t* __restrict__ blen,
                                   int T, int NTB, int NPAD) {
    const int bh = blockIdx.x, d = threadIdx.x;
    float total = 0.f;
    int tokens = 0;
    for (int n = 0; n < NTB; ++n) {
        const size_t o = ((size_t)bh * NPAD + n) * HD + d;
        const float sum = kc[o];
        total += sum;
        const int len = block_len_of(blen, n, T);
        tokens += len;
        kc[o] = sum / (float)len;
    }
    kmean_next[(size_t)bh * HD + d] = total / (float)tokens;
}

// ---- producer-path threshold from the quantized centroid ----
__global__ void prep_thr_from_cen(const int8_t* __restrict__ cen8,
                                  const float* __restrict__ cens,
                                  const float* __restrict__ kcvar,
                                  float* __restrict__ thr,
                                  int NQ, float tau, float log2s) {
    __shared__ float sred[HD];
    const int qb = blockIdx.x, bh = blockIdx.y, d = threadIdx.x;
    const float c = (float)cen8[((size_t)bh * NQ + qb) * HD + perm_d(d)] *
                    cens[(size_t)bh * NQ + qb];
    const float var = block_sum128(c * c * kcvar[(size_t)bh * HD + d], sred);
    if (d == 0) thr[(size_t)bh * NQ + qb] = thr_of(var, tau, log2s);
}

}  // namespace

// Producer path: block K sums left in scratch -> means, pooled quant,
// threshold. The caller places the V scale it quantized with in the vsc slot.
void launch_sol_finish(
    void* scratch, void* kciP, void* kcs, void* threshold,
    const void* cen8, const void* cens, void* kmean_next, const void* blen,
    int B, int T, int H, int NTB, int NPAD, int NQ,
    float tau, float scale_log2, cudaStream_t stream)
{
    const Scratch s = carve_scratch(scratch, B, H, NPAD);
    prep_sums_to_means<<<B * H, HD, 0, stream>>>(
        s.kc, (float*)kmean_next, (const int32_t*)blen, T, NTB, NPAD);
    prep_pooled_stats<<<B * H, HD, 0, stream>>>(
        s.kc, s.kmean, nullptr, nullptr, s.kcvar, NTB, NPAD);
    prep_pooled_quant<<<dim3(NPAD, B * H), HD, 0, stream>>>(
        s.kc, s.kmean, nullptr, (int8_t*)kciP, (float*)kcs, NTB, NPAD);
    prep_thr_from_cen<<<dim3(NQ, B * H), HD, 0, stream>>>(
        (const int8_t*)cen8, (const float*)cens, s.kcvar, (float*)threshold,
        NQ, tau, scale_log2);
}

template <typename E>
static void preprocess_launch(
    const void* q, const void* k, const void* v,
    void* qiP, void* qs, void* kiP, void* ksb, void* kciP, void* kcs,
    void* vcT, void* threshold, void* cen8, void* cens, void* vsc, void* qmean,
    void* scratch,           // sol_preprocess_scratch_bytes
    const void* key_bias,    // [B, T] f32 in log2 units, or nullptr
    const void* blen,        // [NTB] int32 valid tokens per block, or nullptr
    int B, int T, int Tp, int H, int NTB, int NPAD, int NQ,
    int64_t qs_b, int64_t qs_t, int64_t qs_h,
    int64_t ks_b, int64_t ks_t, int64_t ks_h,
    int64_t vs_b, int64_t vs_t, int64_t vs_h,
    float tau, float scale_log2, int qk_balance, cudaStream_t stream)
{
    const Scratch s = carve_scratch(scratch, B, H, NPAD);
    // Off: no balance pass runs and every kernel below takes a null factor,
    // which is the pre-balance arithmetic exactly. On: K's sums of squares
    // ride along in pass 1, Q's take one extra read of q, and the factors
    // feed the pooled, Q and K quantizers. kmean, kcvar, the threshold and
    // qmean stay unbalanced: they are read against each other, never
    // against an int8 carrier.
    const float* fq = qk_balance ? s.fq : nullptr;
    const float* fk = qk_balance ? s.fk : nullptr;

    // pass 1 accumulates V's |max| into vsc by atomicMax
    cudaMemsetAsync(vsc, 0, (size_t)B * H * HD * sizeof(float), stream);
    prep_reduce_kv<E><<<dim3(NPAD, B * H), HD, 0, stream>>>(
        (const E*)k, (const E*)v, s.kc, (__nv_bfloat16*)vcT,
        (float*)vsc, qk_balance ? s.sq : nullptr,
        (const int32_t*)blen, T, H, NTB, NPAD, ks_b, ks_t, ks_h, vs_b, vs_t, vs_h);
    prep_pooled_stats<<<B * H, HD, 0, stream>>>(
        s.kc, s.kmean, (const float*)vsc, (float*)vsc, s.kcvar, NTB, NPAD);
    if (qk_balance) {
        prep_balance_reduce<<<B * H, HD, 0, stream>>>(s.sq, s.fk, NTB, NPAD);   // K^2 sums -> fk (temp)
        prep_reduce_q_sq<E><<<dim3(NQ, B * H), HD, 0, stream>>>(
            (const E*)q, s.sq, (const int32_t*)blen, T, H, NPAD, qs_b, qs_t, qs_h);
        prep_balance_reduce<<<B * H, HD, 0, stream>>>(s.sq, s.fq, NQ, NPAD);    // Q^2 sums -> fq (temp)
        prep_balance_factor<<<B * H, HD, 0, stream>>>(s.fk, s.fq, s.fq, s.fk);
    }
    prep_pooled_quant<<<dim3(NPAD, B * H), HD, 0, stream>>>(
        s.kc, s.kmean, fk, (int8_t*)kciP, (float*)kcs, NTB, NPAD);
    prep_q<E><<<dim3(NQ, B * H), HD, 0, stream>>>(
        (const E*)q, s.kcvar, (int8_t*)qiP, (float*)qs, (float*)threshold,
        (int8_t*)cen8, (float*)cens, (float*)qmean, fq, (const int32_t*)blen,
        T, H, NQ, NPAD, tau, scale_log2, qs_b, qs_t, qs_h);
    prep_k<E><<<dim3(NTB, B * H), HD, 0, stream>>>(
        (const E*)k, s.kmean, (int8_t*)kiP, (float2*)ksb,
        (const float*)key_bias, fk, (const int32_t*)blen, T, Tp, H, ks_b, ks_t, ks_h);
}

void launch_sol_preprocess(
    const void* q, const void* k, const void* v,
    void* qiP, void* qs, void* kiP, void* ksb, void* kciP, void* kcs,
    void* vcT, void* threshold, void* cen8, void* cens, void* vsc, void* qmean,
    void* scratch, const void* key_bias, const void* blen,
    int B, int T, int Tp, int H, int NTB, int NPAD, int NQ,
    int64_t qs_b, int64_t qs_t, int64_t qs_h,
    int64_t ks_b, int64_t ks_t, int64_t ks_h,
    int64_t vs_b, int64_t vs_t, int64_t vs_h,
    float tau, float scale_log2, int elem, int qk_balance, cudaStream_t stream)
{
    auto fn = elem == sol::SOL_FP16 ? preprocess_launch<__half> : preprocess_launch<__nv_bfloat16>;
    fn(q, k, v, qiP, qs, kiP, ksb, kciP, kcs, vcT, threshold, cen8, cens, vsc, qmean,
       scratch, key_bias, blen, B, T, Tp, H, NTB, NPAD, NQ,
       qs_b, qs_t, qs_h, ks_b, ks_t, ks_h, vs_b, vs_t, vs_h, tau, scale_log2, qk_balance, stream);
}

// kc, kmean, kcvar; then the balance's per-block partials (one kc-sized
// array, K's then Q's) and its two per-channel factors. The partials are
// carved whether or not balancing is on: the Plan is sized before the
// call's options are known, and a T^2 index array already sits beside it.
size_t sol_preprocess_scratch_bytes(int B, int H, int NPAD) {
    return ((size_t)2 * B * H * NPAD * HEAD_DIM + (size_t)4 * B * H * HEAD_DIM) * sizeof(float);
}
