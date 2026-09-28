# SPDX-FileCopyrightText: Copyright (c) 2026 Antigravity / Comfy-Kitchen contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

try:
    import pytest
except ImportError:
    class _DummyMark:
        @staticmethod
        def skipif(*args, **kwargs):
            return lambda fn: fn
    class _DummyPytest:
        mark = _DummyMark()
    pytest = _DummyPytest()
import torch

import comfy_kitchen
from comfy_kitchen.backends.eager.rope import rms_rope_pack_kv as eager_rms_rope_pack_kv



def _build_test_tensors(batch=1, seq=1024, prefix=18, heads=32, head_dim=128, dtype=torch.bfloat16, device="cuda"):
    # Target Q, K, V
    q = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, seq, heads, head_dim, dtype=dtype, device=device)

    # Prefix K, V
    k_prefix = torch.randn(batch, prefix, heads, head_dim, dtype=dtype, device=device)
    v_prefix = torch.randn(batch, prefix, heads, head_dim, dtype=dtype, device=device)

    # Scales
    q_scale = torch.randn(head_dim, dtype=dtype, device=device)
    k_scale = torch.randn(head_dim, dtype=dtype, device=device)

    # 6D Rotation matrix for RoPE: [1, seq, 1, head_dim // 2, 2, 2]
    # Representing [[cos, -sin], [sin, cos]]
    angles = torch.randn(1, seq, 1, head_dim // 2, dtype=torch.float32, device=device)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    freqs = torch.stack([
        torch.stack([cos, -sin], dim=-1),
        torch.stack([sin, cos], dim=-1)
    ], dim=-2)  # [1, seq, 1, 64, 2, 2]

    return q, k, v, k_prefix, v_prefix, q_scale, k_scale, freqs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rms_rope_pack_kv_copy_v():
    """Test separate linear projections path (copy_v=True)."""
    batch, seq, prefix, heads, head_dim = 1, 512, 16, 32, 128
    q, k, v, kp, vp, qs, ks, freqs = _build_test_tensors(batch, seq, prefix, heads, head_dim)

    # Pre-allocate output buffers
    k_out_cuda = torch.zeros(batch, prefix + seq, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v_out_cuda = torch.zeros(batch, prefix + seq, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    q_out_cuda = torch.zeros_like(q)

    k_out_eager = torch.zeros_like(k_out_cuda)
    v_out_eager = torch.zeros_like(v_out_cuda)
    q_out_eager = torch.zeros_like(q_out_cuda)

    # Eager reference
    qo_e, ko_e, vo_e = eager_rms_rope_pack_kv(
        q=q.clone(),
        k_out=k_out_eager,
        v_out=v_out_eager,
        freqs_cis=freqs,
        k_prefix=kp,
        v_prefix=vp,
        q_scale=qs,
        k_scale=ks,
        k_src=k,
        v_src=v,
        q_out=q_out_eager,
    )

    # CUDA kernel
    qo_c, ko_c, vo_c = comfy_kitchen.rms_rope_pack_kv(
        q=q.clone(),
        k_out=k_out_cuda,
        v_out=v_out_cuda,
        freqs_cis=freqs,
        k_prefix=kp,
        v_prefix=vp,
        q_scale=qs,
        k_scale=ks,
        k_src=k,
        v_src=v,
        q_out=q_out_cuda,
    )

    # Verify prefix copies
    assert torch.equal(ko_c[:, :prefix], kp), "Prefix K copy mismatch"
    assert torch.equal(vo_c[:, :prefix], vp), "Prefix V copy mismatch"

    # Verify target V copy
    assert torch.equal(vo_c[:, prefix:], v), "Target V copy mismatch"

    # Verify Q and target K norm+RoPE against eager (BF16 rounding tolerance)
    assert torch.allclose(qo_c, qo_e, atol=0.08, rtol=0.05), f"Q mismatch: max diff {(qo_c - qo_e).abs().max()}"
    assert torch.allclose(ko_c[:, prefix:], ko_e[:, prefix:], atol=0.08, rtol=0.05), f"Target K mismatch: max diff {(ko_c[:, prefix:] - ko_e[:, prefix:]).abs().max()}"



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rms_rope_pack_kv_packed_gemm():
    """Test packed GEMM path (k_src=None, v_src=None, target V already in buffer)."""
    batch, seq, prefix, heads, head_dim = 1, 1024, 18, 32, 128
    q, k, v, kp, vp, qs, ks, freqs = _build_test_tensors(batch, seq, prefix, heads, head_dim)

    # In packed GEMM, k and v are already projected into rows [prefix:]
    k_out_cuda = torch.zeros(batch, prefix + seq, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    v_out_cuda = torch.zeros(batch, prefix + seq, heads, head_dim, dtype=torch.bfloat16, device="cuda")
    k_out_cuda[:, prefix:].copy_(k)
    v_out_cuda[:, prefix:].copy_(v)

    k_out_eager = torch.zeros_like(k_out_cuda)
    v_out_eager = torch.zeros_like(v_out_cuda)
    k_out_eager[:, prefix:].copy_(k)
    v_out_eager[:, prefix:].copy_(v)

    q_cuda = q.clone()
    q_eager = q.clone()

    # Eager reference (in-place on q)
    qo_e, ko_e, vo_e = eager_rms_rope_pack_kv(
        q=q_eager,
        k_out=k_out_eager,
        v_out=v_out_eager,
        freqs_cis=freqs,
        k_prefix=kp,
        v_prefix=vp,
        q_scale=qs,
        k_scale=ks,
        k_src=None,
        v_src=None,
        q_out=None,
    )

    # CUDA kernel (in-place on q)
    qo_c, ko_c, vo_c = comfy_kitchen.rms_rope_pack_kv(
        q=q_cuda,
        k_out=k_out_cuda,
        v_out=v_out_cuda,
        freqs_cis=freqs,
        k_prefix=kp,
        v_prefix=vp,
        q_scale=qs,
        k_scale=ks,
        k_src=None,
        v_src=None,
        q_out=None,
    )

    # Verify in-place on q
    assert qo_c is q_cuda

    # Verify prefix copies
    assert torch.equal(ko_c[:, :prefix], kp)
    assert torch.equal(vo_c[:, :prefix], vp)

    # Verify target V was left untouched
    assert torch.equal(vo_c[:, prefix:], v)

    # Verify Q and target K norm+RoPE
    assert torch.allclose(qo_c, qo_e, atol=0.08, rtol=0.05), f"Q mismatch: max diff {(qo_c - qo_e).abs().max()}"
    assert torch.allclose(ko_c[:, prefix:], ko_e[:, prefix:], atol=0.08, rtol=0.05), f"Target K mismatch: max diff {(ko_c[:, prefix:] - ko_e[:, prefix:]).abs().max()}"



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rms_rope_pack_kv_fp16():
    """Test FP16 dtype execution."""
    batch, seq, prefix, heads, head_dim = 1, 512, 16, 32, 128
    q, k, v, kp, vp, qs, ks, freqs = _build_test_tensors(batch, seq, prefix, heads, head_dim, dtype=torch.float16)

    k_out_cuda = torch.zeros(batch, prefix + seq, heads, head_dim, dtype=torch.float16, device="cuda")
    v_out_cuda = torch.zeros(batch, prefix + seq, heads, head_dim, dtype=torch.float16, device="cuda")
    q_out_cuda = torch.zeros_like(q)

    k_out_eager = torch.zeros_like(k_out_cuda)
    v_out_eager = torch.zeros_like(v_out_cuda)
    q_out_eager = torch.zeros_like(q_out_cuda)

    qo_e, ko_e, vo_e = eager_rms_rope_pack_kv(
        q=q.clone(), k_out=k_out_eager, v_out=v_out_eager, freqs_cis=freqs,
        k_prefix=kp, v_prefix=vp, q_scale=qs, k_scale=ks, k_src=k, v_src=v, q_out=q_out_eager,
    )
    qo_c, ko_c, vo_c = comfy_kitchen.rms_rope_pack_kv(
        q=q.clone(), k_out=k_out_cuda, v_out=v_out_cuda, freqs_cis=freqs,
        k_prefix=kp, v_prefix=vp, q_scale=qs, k_scale=ks, k_src=k, v_src=v, q_out=q_out_cuda,
    )

    assert torch.equal(ko_c[:, :prefix], kp)
    assert torch.equal(vo_c[:, :prefix], vp)
    assert torch.equal(vo_c[:, prefix:], v)
    assert torch.allclose(qo_c, qo_e, atol=1e-2, rtol=1e-2), f"FP16 Q mismatch: max diff {(qo_c - qo_e).abs().max()}"
    assert torch.allclose(ko_c[:, prefix:], ko_e[:, prefix:], atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    print("Running test_rms_rope_pack_kv_copy_v...")
    test_rms_rope_pack_kv_copy_v()
    print("test_rms_rope_pack_kv_copy_v PASSED!")
    print("Running test_rms_rope_pack_kv_packed_gemm...")
    test_rms_rope_pack_kv_packed_gemm()
    print("test_rms_rope_pack_kv_packed_gemm PASSED!")
    print("Running test_rms_rope_pack_kv_fp16...")
    test_rms_rope_pack_kv_fp16()
    print("test_rms_rope_pack_kv_fp16 PASSED!")
    print("ALL TESTS PASSED SUCCESSFULLY!")


