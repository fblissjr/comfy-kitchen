/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Antigravity / Comfy-Kitchen contributors. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dtype_dispatch.cuh"
#include "rope_device.cuh"
#include "utils.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <algorithm>

namespace comfy {
namespace {

constexpr int kHeadDim = 128;
constexpr int kThreadsPerWarpConst = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * kThreadsPerWarpConst;
constexpr int64_t kMaxBlocks = 65535;

template <typename T>
__device__ __forceinline__ void norm_rope_row_128(
    const T* __restrict__ src,
    T* __restrict__ dst,
    const T* __restrict__ weight,
    const float* __restrict__ freqs_base,
    int64_t f_s3, int64_t f_s4, int64_t f_s5,
    float epsilon,
    int lane) {
  // Each lane owns 4 consecutive elements: [4*lane, 4*lane + 3]
  T x[4];
  T w[4];
  *reinterpret_cast<uint64_t*>(x) = *reinterpret_cast<const uint64_t*>(src + lane * 4);
  *reinterpret_cast<uint64_t*>(w) = *reinterpret_cast<const uint64_t*>(weight + lane * 4);

  float val[4];
  val[0] = static_cast<float>(x[0]);
  val[1] = static_cast<float>(x[1]);
  val[2] = static_cast<float>(x[2]);
  val[3] = static_cast<float>(x[3]);

  // Sum of squares across the 4 elements in lane
  float part = fmaf(val[0], val[0], 0.0f);
  part = fmaf(val[1], val[1], part);
  part = fmaf(val[2], val[2], part);
  part = fmaf(val[3], val[3], part);

  // 32-lane warp reduction
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    part += __shfl_xor_sync(0xffffffffu, part, offset);
  }

  const float variance = part * (1.0f / 128.0f);
  const float inv = rsqrtf(variance + epsilon);

  // Double rounding matching diffusers/PyTorch eager contract (cast_x_before_out_mul=True):
  // Round to T after normalize, then round to T after weight multiply.
  float out_f32[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float normalized = static_cast<float>(static_cast<T>(val[i] * inv));
    out_f32[i] = static_cast<float>(static_cast<T>(normalized * static_cast<float>(w[i])));
  }

  // Interleaved RoPE on two pairs: (0, 1) and (2, 3)
  const int p0 = 2 * lane;
  const int p1 = 2 * lane + 1;

  const float f00_0 = freqs_base[p0 * f_s3];
  const float f01_0 = freqs_base[p0 * f_s3 + f_s5];
  const float f10_0 = freqs_base[p0 * f_s3 + f_s4];
  const float f11_0 = freqs_base[p0 * f_s3 + f_s4 + f_s5];

  const float f00_1 = freqs_base[p1 * f_s3];
  const float f01_1 = freqs_base[p1 * f_s3 + f_s5];
  const float f10_1 = freqs_base[p1 * f_s3 + f_s4];
  const float f11_1 = freqs_base[p1 * f_s3 + f_s4 + f_s5];

  T out[4];
  out[0] = static_cast<T>(fmaf(out_f32[0], f00_0, out_f32[1] * f01_0));
  out[1] = static_cast<T>(fmaf(out_f32[0], f10_0, out_f32[1] * f11_0));
  out[2] = static_cast<T>(fmaf(out_f32[2], f00_1, out_f32[3] * f01_1));
  out[3] = static_cast<T>(fmaf(out_f32[2], f10_1, out_f32[3] * f11_1));

  *reinterpret_cast<uint64_t*>(dst + lane * 4) = *reinterpret_cast<const uint64_t*>(out);
}

template <typename T>
__device__ __forceinline__ void copy_row_128(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int lane) {
  *reinterpret_cast<uint64_t*>(dst + lane * 4) = *reinterpret_cast<const uint64_t*>(src + lane * 4);
}

template <typename T, bool kCopyV>
__global__ __launch_bounds__(kThreads) void rms_rope_pack_kernel(
    const T* __restrict__ q,
    const T* __restrict__ k_src,
    const T* __restrict__ v_src,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    T* __restrict__ v_out,
    const T* __restrict__ k_prefix,
    const T* __restrict__ v_prefix,
    const float* __restrict__ freqs,
    const T* __restrict__ q_scale,
    const T* __restrict__ k_scale,
    int64_t batch,
    int64_t seq,
    int64_t prefix,
    int64_t heads,
    int64_t q_s0, int64_t q_s1, int64_t q_s2,
    int64_t qo_s0, int64_t qo_s1, int64_t qo_s2,
    int64_t ks_s0, int64_t ks_s1, int64_t ks_s2,
    int64_t ko_s0, int64_t ko_s1, int64_t ko_s2,
    int64_t vs_s0, int64_t vs_s1, int64_t vs_s2,
    int64_t vo_s0, int64_t vo_s1, int64_t vo_s2,
    int64_t kp_s0, int64_t kp_s1, int64_t kp_s2,
    int64_t vp_s0, int64_t vp_s1, int64_t vp_s2,
    int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3, int64_t f_s4, int64_t f_s5,
    int64_t freqs_batch, int64_t freqs_heads,
    float epsilon) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  const int64_t rows_q = batch * seq * heads;
  const int64_t rows_k = rows_q;
  const int64_t rows_pk = batch * prefix * heads;
  const int64_t rows_pv = rows_pk;
  const int64_t rows_v = kCopyV ? rows_q : 0;
  const int64_t total = rows_q + rows_k + rows_pk + rows_pv + rows_v;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kWarpsPerBlock;

  for (int64_t row = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
       row < total; row += stride) {
    int64_t r = row;
    if (r < rows_q) {
      // Q row: normalize and rotate
      const int64_t h = r % heads;
      const int64_t t = r / heads;
      const int64_t s = t % seq;
      const int64_t b = t / seq;

      const T* src_ptr = q + b * q_s0 + s * q_s1 + h * q_s2;
      T* dst_ptr = q_out + b * qo_s0 + s * qo_s1 + h * qo_s2;
      const int64_t fb = freqs_batch == 1 ? 0 : b;
      const int64_t fh = freqs_heads == 1 ? 0 : h;
      const float* f_base = freqs + fb * f_s0 + s * f_s1 + fh * f_s2;

      norm_rope_row_128(src_ptr, dst_ptr, q_scale, f_base, f_s3, f_s4, f_s5, epsilon, lane);
      continue;
    }
    r -= rows_q;
    if (r < rows_k) {
      // Target K row: normalize and rotate into k_out rows [prefix + s]
      const int64_t h = r % heads;
      const int64_t t = r / heads;
      const int64_t s = t % seq;
      const int64_t b = t / seq;

      const T* src_ptr = (k_src != nullptr)
                             ? (k_src + b * ks_s0 + s * ks_s1 + h * ks_s2)
                             : (k_out + b * ko_s0 + (prefix + s) * ko_s1 + h * ko_s2);
      T* dst_ptr = k_out + b * ko_s0 + (prefix + s) * ko_s1 + h * ko_s2;
      const int64_t fb = freqs_batch == 1 ? 0 : b;
      const int64_t fh = freqs_heads == 1 ? 0 : h;
      const float* f_base = freqs + fb * f_s0 + s * f_s1 + fh * f_s2;

      norm_rope_row_128(src_ptr, dst_ptr, k_scale, f_base, f_s3, f_s4, f_s5, epsilon, lane);
      continue;
    }
    r -= rows_k;
    if (r < rows_pk) {
      // Prefix K row: copy into k_out rows [0 : prefix]
      const int64_t h = r % heads;
      const int64_t t = r / heads;
      const int64_t p = t % prefix;
      const int64_t b = t / prefix;

      const T* src_ptr = k_prefix + b * kp_s0 + p * kp_s1 + h * kp_s2;
      T* dst_ptr = k_out + b * ko_s0 + p * ko_s1 + h * ko_s2;
      copy_row_128(src_ptr, dst_ptr, lane);
      continue;
    }
    r -= rows_pk;
    if (r < rows_pv) {
      // Prefix V row: copy into v_out rows [0 : prefix]
      const int64_t h = r % heads;
      const int64_t t = r / heads;
      const int64_t p = t % prefix;
      const int64_t b = t / prefix;

      const T* src_ptr = v_prefix + b * vp_s0 + p * vp_s1 + h * vp_s2;
      T* dst_ptr = v_out + b * vo_s0 + p * vo_s1 + h * vo_s2;
      copy_row_128(src_ptr, dst_ptr, lane);
      continue;
    }
    if constexpr (kCopyV) {
      r -= rows_pv;
      if (r < rows_v) {
        // Target V row: copy from v_src into v_out rows [prefix + s]
        const int64_t h = r % heads;
        const int64_t t = r / heads;
        const int64_t s = t % seq;
        const int64_t b = t / seq;

        const T* src_ptr = v_src + b * vs_s0 + s * vs_s1 + h * vs_s2;
        T* dst_ptr = v_out + b * vo_s0 + (prefix + s) * vo_s1 + h * vo_s2;
        copy_row_128(src_ptr, dst_ptr, lane);
      }
    }
  }
}

} // namespace
} // namespace comfy

extern "C" {

void launch_rms_rope_pack_kernel(
    const void* q,
    const void* k_src,
    const void* v_src,
    void* q_out,
    void* k_out,
    void* v_out,
    const void* k_prefix,
    const void* v_prefix,
    const void* freqs,
    const void* q_scale,
    const void* k_scale,
    int64_t batch,
    int64_t seq,
    int64_t prefix,
    int64_t heads,
    int64_t head_dim,
    int64_t q_s0, int64_t q_s1, int64_t q_s2,
    int64_t qo_s0, int64_t qo_s1, int64_t qo_s2,
    int64_t ks_s0, int64_t ks_s1, int64_t ks_s2,
    int64_t ko_s0, int64_t ko_s1, int64_t ko_s2,
    int64_t vs_s0, int64_t vs_s1, int64_t vs_s2,
    int64_t vo_s0, int64_t vo_s1, int64_t vo_s2,
    int64_t kp_s0, int64_t kp_s1, int64_t kp_s2,
    int64_t vp_s0, int64_t vp_s1, int64_t vp_s2,
    int64_t f_s0, int64_t f_s1, int64_t f_s2, int64_t f_s3, int64_t f_s4, int64_t f_s5,
    int64_t freqs_batch, int64_t freqs_heads,
    float epsilon,
    bool copy_v,
    int dtype_code,
    cudaStream_t stream) {
  if (batch == 0 || seq == 0 || heads == 0 || head_dim != comfy::kHeadDim) {
    return;
  }

  const int64_t total_rows = batch * heads * (2 * seq + 2 * prefix + (copy_v ? seq : 0));
  const int64_t needed_blocks = (total_rows + comfy::kWarpsPerBlock - 1) / comfy::kWarpsPerBlock;
  const uint32_t blocks = static_cast<uint32_t>(std::min<int64_t>(needed_blocks, comfy::kMaxBlocks));

  if (dtype_code == 2) { // bfloat16
    using T = __nv_bfloat16;
    if (copy_v) {
      comfy::rms_rope_pack_kernel<T, true><<<blocks, comfy::kThreads, 0, stream>>>(
          static_cast<const T*>(q),
          static_cast<const T*>(k_src),
          static_cast<const T*>(v_src),
          static_cast<T*>(q_out),
          static_cast<T*>(k_out),
          static_cast<T*>(v_out),
          static_cast<const T*>(k_prefix),
          static_cast<const T*>(v_prefix),
          static_cast<const float*>(freqs),
          static_cast<const T*>(q_scale),
          static_cast<const T*>(k_scale),
          batch, seq, prefix, heads,
          q_s0, q_s1, q_s2,
          qo_s0, qo_s1, qo_s2,
          ks_s0, ks_s1, ks_s2,
          ko_s0, ko_s1, ko_s2,
          vs_s0, vs_s1, vs_s2,
          vo_s0, vo_s1, vo_s2,
          kp_s0, kp_s1, kp_s2,
          vp_s0, vp_s1, vp_s2,
          f_s0, f_s1, f_s2, f_s3, f_s4, f_s5,
          freqs_batch, freqs_heads,
          epsilon);
    } else {
      comfy::rms_rope_pack_kernel<T, false><<<blocks, comfy::kThreads, 0, stream>>>(
          static_cast<const T*>(q),
          static_cast<const T*>(k_src),
          static_cast<const T*>(v_src),
          static_cast<T*>(q_out),
          static_cast<T*>(k_out),
          static_cast<T*>(v_out),
          static_cast<const T*>(k_prefix),
          static_cast<const T*>(v_prefix),
          static_cast<const float*>(freqs),
          static_cast<const T*>(q_scale),
          static_cast<const T*>(k_scale),
          batch, seq, prefix, heads,
          q_s0, q_s1, q_s2,
          qo_s0, qo_s1, qo_s2,
          ks_s0, ks_s1, ks_s2,
          ko_s0, ko_s1, ko_s2,
          vs_s0, vs_s1, vs_s2,
          vo_s0, vo_s1, vo_s2,
          kp_s0, kp_s1, kp_s2,
          vp_s0, vp_s1, vp_s2,
          f_s0, f_s1, f_s2, f_s3, f_s4, f_s5,
          freqs_batch, freqs_heads,
          epsilon);
    }
  } else if (dtype_code == 1) { // float16
    using T = half;
    if (copy_v) {
      comfy::rms_rope_pack_kernel<T, true><<<blocks, comfy::kThreads, 0, stream>>>(
          static_cast<const T*>(q),
          static_cast<const T*>(k_src),
          static_cast<const T*>(v_src),
          static_cast<T*>(q_out),
          static_cast<T*>(k_out),
          static_cast<T*>(v_out),
          static_cast<const T*>(k_prefix),
          static_cast<const T*>(v_prefix),
          static_cast<const float*>(freqs),
          static_cast<const T*>(q_scale),
          static_cast<const T*>(k_scale),
          batch, seq, prefix, heads,
          q_s0, q_s1, q_s2,
          qo_s0, qo_s1, qo_s2,
          ks_s0, ks_s1, ks_s2,
          ko_s0, ko_s1, ko_s2,
          vs_s0, vs_s1, vs_s2,
          vo_s0, vo_s1, vo_s2,
          kp_s0, kp_s1, kp_s2,
          vp_s0, vp_s1, vp_s2,
          f_s0, f_s1, f_s2, f_s3, f_s4, f_s5,
          freqs_batch, freqs_heads,
          epsilon);
    } else {
      comfy::rms_rope_pack_kernel<T, false><<<blocks, comfy::kThreads, 0, stream>>>(
          static_cast<const T*>(q),
          static_cast<const T*>(k_src),
          static_cast<const T*>(v_src),
          static_cast<T*>(q_out),
          static_cast<T*>(k_out),
          static_cast<T*>(v_out),
          static_cast<const T*>(k_prefix),
          static_cast<const T*>(v_prefix),
          static_cast<const float*>(freqs),
          static_cast<const T*>(q_scale),
          static_cast<const T*>(k_scale),
          batch, seq, prefix, heads,
          q_s0, q_s1, q_s2,
          qo_s0, qo_s1, qo_s2,
          ks_s0, ks_s1, ks_s2,
          ko_s0, ko_s1, ko_s2,
          vs_s0, vs_s1, vs_s2,
          vo_s0, vo_s1, vo_s2,
          kp_s0, kp_s1, kp_s2,
          vp_s0, vp_s1, vp_s2,
          f_s0, f_s1, f_s2, f_s3, f_s4, f_s5,
          freqs_batch, freqs_heads,
          epsilon);
    }
  }
}

} // extern "C"
