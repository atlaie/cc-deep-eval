"""
Local unit tests for `compute_attention_stats`.

Designed to run in <1s on CPU with `pytest test_attention_stats.py -v`.
No vLLM, no GPU, no model load. Covers the five mandatory smoke-check
categories from EXTENSION_PHASE2.md §11 (shape, dtype, value-range,
variation across prompts, determinism on the first causal position),
a closed-form known-answer test (uniform attention via zeroed weights),
and Layout A (fused_qkv_a_proj) coverage matching the GLM 5.1 / Tinfoil
DeepSeek V2 fork code path that runs in production.

Drop next to `attention_stats.py` and run from that directory, or set
PYTHONPATH so both modules import.
"""

import math

import pytest
import torch
import torch.nn as nn

from vllm_lens.attention_stats import compute_attention_stats


# ── Test fixtures ────────────────────────────────────────────────────────────


class FakeReplicatedLinear(nn.Module):
    """Mimics vLLM's ReplicatedLinear/ColumnParallelLinear: forward returns (out, bias).

    Real production code unpacks `out[0]` to get the tensor; the compute function
    must do this too. Testing with a tuple-returning module catches the
    AttributeError-on-tuple bug that already cost a worker reload in Phase 1
    (per PHASE2_PLAN.md §2.5).
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor):
        return self.linear(x), None


class FakeRMSNorm(nn.Module):
    """vLLM uses RMSNorm (not LayerNorm) for `q_a_layernorm` and `kv_a_layernorm`."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class FakeMLAAttention(nn.Module):
    """Layout B (standard separate LoRA): minimal DeepseekV2MLAAttention
    shape-twin. Dimensions much smaller than GLM 5.1 but interface identical
    so the compute function can't tell them apart."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_local_heads: int = 4,
        qk_nope_head_dim: int = 16,
        qk_rope_head_dim: int = 8,
        v_head_dim: int = 16,
        kv_lora_rank: int = 32,
        q_lora_rank: int = 48,
    ):
        super().__init__()
        self.num_local_heads = num_local_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank

        # Layout B: separate Q LoRA + KV-with-MQA projections.
        self.q_a_proj = FakeReplicatedLinear(hidden_size, q_lora_rank)
        self.q_a_layernorm = FakeRMSNorm(q_lora_rank)
        self.q_b_proj = FakeReplicatedLinear(
            q_lora_rank, num_local_heads * self.qk_head_dim
        )
        self.kv_a_proj_with_mqa = FakeReplicatedLinear(
            hidden_size, kv_lora_rank + qk_rope_head_dim
        )
        self.kv_a_layernorm = FakeRMSNorm(kv_lora_rank)
        self.kv_b_proj = FakeReplicatedLinear(
            kv_lora_rank, num_local_heads * (qk_nope_head_dim + v_head_dim)
        )


class FakeFusedMLAAttention(nn.Module):
    """Layout A (fused Q+KV-A projection): GLM 5.1 / Tinfoil DeepSeek V2 fork.

    The fused_qkv_a_proj outputs (q_lora_rank + kv_lora_rank + qk_rope_head_dim)
    concatenated; compute_attention_stats must split at q_lora_rank rather
    than calling separate q_a_proj / kv_a_proj_with_mqa attributes (which
    don't exist on this layout).

    Dimensions deliberately match GLM 5.1's MLA block (H=8, qk_nope=192,
    kv_lora=512, q_lora=2048, hidden=6144) so the fixture exercises the same
    arithmetic the real model does.
    """

    def __init__(
        self,
        hidden_size: int = 6144,
        num_local_heads: int = 8,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 2048,
    ):
        super().__init__()
        self.num_local_heads = num_local_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank

        # Layout A: fused projection.
        # Output is [q_lora_rank | kv_lora_rank + qk_rope_head_dim], split at q_lora_rank.
        self.fused_qkv_a_proj = FakeReplicatedLinear(
            hidden_size, q_lora_rank + kv_lora_rank + qk_rope_head_dim
        )
        self.q_a_layernorm = FakeRMSNorm(q_lora_rank)
        self.q_b_proj = FakeReplicatedLinear(
            q_lora_rank, num_local_heads * self.qk_head_dim
        )
        self.kv_a_layernorm = FakeRMSNorm(kv_lora_rank)
        self.kv_b_proj = FakeReplicatedLinear(
            kv_lora_rank, num_local_heads * (qk_nope_head_dim + v_head_dim)
        )
        # Note: NO q_a_proj, NO kv_a_proj_with_mqa — they are absent on this
        # layout and compute_attention_stats must not reference them.


@pytest.fixture
def mod():
    """Default fake attention module (Layout B), deterministic init."""
    torch.manual_seed(0)
    return FakeMLAAttention()


@pytest.fixture
def hs():
    """Default hidden_states (T=20, hidden=64)."""
    torch.manual_seed(1)
    return torch.randn(20, 64)


@pytest.fixture
def fused_mod():
    """Fused-layout attention module sized to GLM 5.1's MLA block."""
    torch.manual_seed(0)
    return FakeFusedMLAAttention()


@pytest.fixture
def fused_hs():
    """hidden_states matching FakeFusedMLAAttention's hidden_size (6144)."""
    torch.manual_seed(1)
    return torch.randn(8, 6144)


# ── Smoke-check 1: Shape ─────────────────────────────────────────────────────


def test_shape_default(mod, hs):
    out = compute_attention_stats(mod, hs)
    T = hs.shape[0]
    H = mod.num_local_heads
    assert out["entropy"].shape == (H, T)
    assert out["rowmax"].shape == (H, T)
    assert out["top10_mass"].shape == (H, T)


@pytest.mark.parametrize("T", [1, 2, 10, 11, 100])
def test_shape_varied_seqlen(mod, T):
    """Edge cases: T=1 (decode-only), T=10 (top-k boundary), T=11, T=100."""
    hs = torch.randn(T, 64)
    out = compute_attention_stats(mod, hs)
    H = mod.num_local_heads
    for k, v in out.items():
        assert v.shape == (H, T), f"{k}: got {v.shape}, want {(H, T)}"


# ── Smoke-check 2: Dtype ─────────────────────────────────────────────────────


def test_dtype_fp32_inputs(mod, hs):
    out = compute_attention_stats(mod, hs)
    for name, v in out.items():
        assert v.dtype == torch.float32, f"{name}: {v.dtype}"


def test_dtype_bf16_inputs():
    """Production activations are bf16. Module should also be bf16. Outputs stay fp32."""
    torch.manual_seed(0)
    mod = FakeMLAAttention().bfloat16()
    hs = torch.randn(20, 64).bfloat16()
    out = compute_attention_stats(mod, hs)
    for name, v in out.items():
        assert v.dtype == torch.float32, f"{name}: {v.dtype}"
        assert torch.isfinite(v).all(), f"{name}: contains NaN/Inf at bf16"


# ── Smoke-check 3: Value range ───────────────────────────────────────────────


def test_finite(mod, hs):
    out = compute_attention_stats(mod, hs)
    for name, v in out.items():
        assert torch.isfinite(v).all(), f"{name} has NaN or Inf"


def test_rowmax_in_unit_interval(mod, hs):
    out = compute_attention_stats(mod, hs)
    assert (out["rowmax"] >= 0).all()
    assert (out["rowmax"] <= 1.0 + 1e-5).all()


def test_top10_in_unit_interval(mod, hs):
    out = compute_attention_stats(mod, hs)
    assert (out["top10_mass"] >= 0).all()
    assert (out["top10_mass"] <= 1.0 + 1e-5).all()


def test_entropy_in_valid_range(mod, hs):
    """Entropy in bits is bounded above by log2(T) (uniform distribution).
    For row i under causal masking, the tighter bound is log2(i+1)."""
    T = hs.shape[0]
    out = compute_attention_stats(mod, hs)
    assert (out["entropy"] >= -1e-5).all()
    assert (out["entropy"] <= math.log2(T) + 1e-5).all()


def test_top10_geq_rowmax(mod, hs):
    """Top-10% mass must dominate the single largest weight."""
    out = compute_attention_stats(mod, hs)
    assert (out["top10_mass"] >= out["rowmax"] - 1e-5).all()


# ── Smoke-check 4: Variation across prompts ──────────────────────────────────


def test_variation_across_inputs(mod):
    """Distinct hidden_states should produce distinct stats at token 1+."""
    torch.manual_seed(0)
    hs1 = torch.randn(20, 64)
    hs2 = torch.randn(20, 64)
    out1 = compute_attention_stats(mod, hs1)
    out2 = compute_attention_stats(mod, hs2)
    assert not torch.allclose(out1["entropy"][:, 1], out2["entropy"][:, 1], atol=1e-4)
    assert not torch.allclose(out1["rowmax"][:, 1], out2["rowmax"][:, 1], atol=1e-4)


# ── Smoke-check 5: First-position determinism under causal mask ──────────────


def test_first_token_causal_collapse(mod, hs):
    """Under causal masking, token 0 attends only to itself: entropy=0, max=1."""
    out = compute_attention_stats(mod, hs)
    H = mod.num_local_heads
    assert torch.allclose(out["entropy"][:, 0], torch.zeros(H), atol=1e-5)
    assert torch.allclose(out["rowmax"][:, 0], torch.ones(H), atol=1e-5)
    assert torch.allclose(out["top10_mass"][:, 0], torch.ones(H), atol=1e-5)


# ── Closed-form known-answer test ────────────────────────────────────────────


def test_uniform_attention_when_projections_zero():
    """Force Q_nope = K_nope = 0 by zeroing all projection weights. Then
    scores = 0 everywhere, softmax over the causal mask gives uniform
    distribution: entropy[h, i] = log2(i+1) exactly. Catches scaling and
    masking bugs that variation/range checks would miss.
    """
    torch.manual_seed(0)
    mod = FakeMLAAttention()
    for m in mod.modules():
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
    for m in [mod.q_a_layernorm, mod.kv_a_layernorm]:
        nn.init.zeros_(m.weight)

    T = 16
    hs = torch.randn(T, 64)
    out = compute_attention_stats(mod, hs)

    expected = torch.log2(torch.arange(1, T + 1, dtype=torch.float32))
    for h in range(mod.num_local_heads):
        torch.testing.assert_close(out["entropy"][h], expected, atol=1e-4, rtol=1e-4)

    expected_max = 1.0 / torch.arange(1, T + 1, dtype=torch.float32)
    for h in range(mod.num_local_heads):
        torch.testing.assert_close(out["rowmax"][h], expected_max, atol=1e-5, rtol=1e-4)


def test_t_equals_one_decode_edge():
    """Decode-step edge: T=1. Single token, single position, mass=1, entropy=0."""
    torch.manual_seed(0)
    mod = FakeMLAAttention()
    hs = torch.randn(1, 64)
    out = compute_attention_stats(mod, hs)
    assert out["entropy"].shape == (mod.num_local_heads, 1)
    torch.testing.assert_close(
        out["entropy"], torch.zeros(mod.num_local_heads, 1), atol=1e-5, rtol=0.0
    )
    torch.testing.assert_close(
        out["rowmax"], torch.ones(mod.num_local_heads, 1), atol=1e-5, rtol=0.0
    )


def test_non_causal_no_collapse():
    """With causal=False, token 0 attends to all positions — entropy should be > 0."""
    torch.manual_seed(0)
    mod = FakeMLAAttention()
    hs = torch.randn(20, 64)
    out = compute_attention_stats(mod, hs, causal=False)
    assert (out["entropy"][:, 0] > 0).any()


# ── Sanity: input validation ─────────────────────────────────────────────────


def test_rejects_3d_input(mod):
    bad = torch.randn(1, 20, 64)
    with pytest.raises(ValueError, match="must be"):
        compute_attention_stats(mod, bad)


# ── Layout A (fused_qkv_a_proj): GLM 5.1 / Tinfoil DeepSeek V2 fork ──────────
#
# These tests exercise the production code path. Layout B (above) catches
# logic bugs; Layout A catches layout-dispatch bugs that would only surface
# at deploy time on the real model.


def test_fused_layout_shapes(fused_mod, fused_hs):
    """Smoke: fused layout produces correctly-shaped outputs."""
    out = compute_attention_stats(fused_mod, fused_hs)
    T = fused_hs.shape[0]
    H = fused_mod.num_local_heads
    for k, v in out.items():
        assert v.shape == (H, T), f"{k}: got {v.shape}, want {(H, T)}"


def test_fused_layout_finite(fused_mod, fused_hs):
    """At GLM 5.1's dims (qk_nope=192 → 1/sqrt(192) scaling), scores stay numerically
    well-behaved through the softmax."""
    out = compute_attention_stats(fused_mod, fused_hs)
    for name, v in out.items():
        assert torch.isfinite(v).all(), f"{name} has NaN or Inf at production dims"


def test_fused_layout_value_ranges(fused_mod, fused_hs):
    """Same range invariants as Layout B."""
    T = fused_hs.shape[0]
    out = compute_attention_stats(fused_mod, fused_hs)
    assert (out["rowmax"] >= 0).all()
    assert (out["rowmax"] <= 1.0 + 1e-5).all()
    assert (out["top10_mass"] >= 0).all()
    assert (out["top10_mass"] <= 1.0 + 1e-5).all()
    assert (out["top10_mass"] >= out["rowmax"] - 1e-5).all()
    assert (out["entropy"] >= -1e-5).all()
    assert (out["entropy"] <= math.log2(T) + 1e-5).all()


def test_fused_layout_first_token_causal_collapse(fused_mod, fused_hs):
    """Causal mask collapse must work on the fused path too."""
    out = compute_attention_stats(fused_mod, fused_hs)
    H = fused_mod.num_local_heads
    assert torch.allclose(out["entropy"][:, 0], torch.zeros(H), atol=1e-5)
    assert torch.allclose(out["rowmax"][:, 0], torch.ones(H), atol=1e-5)


def test_fused_layout_no_layout_b_attrs(fused_mod):
    """Sanity: fixture genuinely lacks Layout B attributes. If this fails, the
    fixture has drifted and the Layout A path isn't actually being exercised."""
    assert not hasattr(fused_mod, "q_a_proj"), (
        "FakeFusedMLAAttention must not have q_a_proj — otherwise Layout B "
        "would be picked by hasattr-dispatch and this whole test class would "
        "silently exercise the wrong path."
    )
    assert not hasattr(fused_mod, "kv_a_proj_with_mqa")
    # And confirm the fused attribute IS present.
    assert hasattr(fused_mod, "fused_qkv_a_proj")


def test_fused_layout_uniform_when_projections_zero():
    """Closed-form check on Layout A: zero all linear weights AND RMSNorm scales,
    so Q_nope = K_nope = 0 → scores = 0 → uniform softmax under causal mask →
    entropy[i] = log2(i+1) exactly."""
    torch.manual_seed(0)
    mod = FakeFusedMLAAttention(
        hidden_size=64,
        num_local_heads=4,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        kv_lora_rank=32,
        q_lora_rank=48,
    )  # smaller dims so the test runs fast
    for m in mod.modules():
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
    for ln in [mod.q_a_layernorm, mod.kv_a_layernorm]:
        nn.init.zeros_(ln.weight)

    T = 16
    hs = torch.randn(T, 64)
    out = compute_attention_stats(mod, hs)

    expected = torch.log2(torch.arange(1, T + 1, dtype=torch.float32))
    for h in range(mod.num_local_heads):
        torch.testing.assert_close(out["entropy"][h], expected, atol=1e-4, rtol=1e-4)


def test_fused_and_standard_layouts_agree_when_initialized_equivalently():
    """When the fused projection is constructed so its output is the column-stacked
    concatenation of an equivalent (q_a_proj, kv_a_proj_with_mqa) pair, the two
    layouts must produce numerically identical stats. This is the strongest
    cross-check that the layout dispatch is faithful and not silently rewiring
    something on the fused path."""
    torch.manual_seed(0)

    hidden = 64
    H = 4
    d_nope = 16
    d_pe = 8
    V = 16
    L = 32
    qlr = 48

    mod_b = FakeMLAAttention(
        hidden_size=hidden,
        num_local_heads=H,
        qk_nope_head_dim=d_nope,
        qk_rope_head_dim=d_pe,
        v_head_dim=V,
        kv_lora_rank=L,
        q_lora_rank=qlr,
    )
    mod_a = FakeFusedMLAAttention(
        hidden_size=hidden,
        num_local_heads=H,
        qk_nope_head_dim=d_nope,
        qk_rope_head_dim=d_pe,
        v_head_dim=V,
        kv_lora_rank=L,
        q_lora_rank=qlr,
    )

    # Stitch the fused weight to match (q_a_proj || kv_a_proj_with_mqa).
    with torch.no_grad():
        wq = mod_b.q_a_proj.linear.weight              # (qlr, hidden)
        wkv = mod_b.kv_a_proj_with_mqa.linear.weight   # (L + d_pe, hidden)
        mod_a.fused_qkv_a_proj.linear.weight.copy_(torch.cat([wq, wkv], dim=0))
        # Share the remaining params so the two paths really are equivalent.
        mod_a.q_a_layernorm.weight.copy_(mod_b.q_a_layernorm.weight)
        mod_a.q_b_proj.linear.weight.copy_(mod_b.q_b_proj.linear.weight)
        mod_a.kv_a_layernorm.weight.copy_(mod_b.kv_a_layernorm.weight)
        mod_a.kv_b_proj.linear.weight.copy_(mod_b.kv_b_proj.linear.weight)

    hs = torch.randn(20, hidden)
    out_b = compute_attention_stats(mod_b, hs)
    out_a = compute_attention_stats(mod_a, hs)

    for k in ("entropy", "rowmax", "top10_mass"):
        torch.testing.assert_close(
            out_a[k], out_b[k], atol=1e-5, rtol=1e-4,
            msg=f"Layout A and B disagree on {k}",
        )
