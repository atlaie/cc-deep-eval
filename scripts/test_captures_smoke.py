"""Smoke test of captures.py: stubs out network-bound deps and verifies the
pure-logic functions (asserts, entropy diagnostics, NPZ save) work on
synthetic data."""
from __future__ import annotations

import sys
import types
import tempfile
from pathlib import Path

# ---- stub out zstandard / openai / datasets so `import captures` works -----
sys.modules.setdefault("zstandard", types.ModuleType("zstandard"))
sys.modules["zstandard"].ZstdDecompressor = lambda *a, **k: None  # not exercised here

openai_mod = types.ModuleType("openai")
class _FakeOpenAI:
    def __init__(self, *a, **k): pass
openai_mod.OpenAI = _FakeOpenAI
sys.modules["openai"] = openai_mod

datasets_mod = types.ModuleType("datasets")
datasets_mod.load_dataset = lambda *a, **k: None
sys.modules["datasets"] = datasets_mod

# ---- now we can import captures cleanly ------------------------------------
import numpy as np
import captures as C

print("=== test 1: dataclass round-trip ===")
req = C.CaptureRequest(
    residual_layers=[12, 23],
    routing_layers=None,
    attention_layers=[12, 23],
)
xargs = req.to_xargs()
assert xargs == {
    "output_residual_stream": [12, 23],
    "output_attention_stats": [12, 23],
}, xargs
assert req.any_capture()
empty = C.CaptureRequest()
assert empty.to_xargs() == {}
assert not empty.any_capture()
print("  PASS")

print("=== test 2: routing value-range asserts ===")
# Valid case
good = {
    "topk_ids": np.random.randint(0, 256, size=(75, 16, 8)).astype(np.int16),
    "topk_weights": np.random.rand(75, 16, 8).astype(np.float32),
}
C.assert_routing_valid(good)
print("  good payload: PASS")

# Out-of-range expert id
bad = dict(good)
bad["topk_ids"] = good["topk_ids"].copy()
bad["topk_ids"][0, 0, 0] = 999
try:
    C.assert_routing_valid(bad)
    print("  FAIL: should have raised on id=999"); sys.exit(1)
except C.ValueRangeError as e:
    print(f"  caught: {e}")

# Wrong top_k
bad2 = {"topk_ids": np.zeros((75, 16, 4), dtype=np.int16),
        "topk_weights": np.zeros((75, 16, 4), dtype=np.float32)}
try:
    C.assert_routing_valid(bad2)
    print("  FAIL: should have raised on top_k=4"); sys.exit(1)
except C.ValueRangeError as e:
    print(f"  caught: {e}")

# Weights out of [0,1]
bad3 = dict(good)
bad3["topk_weights"] = good["topk_weights"].copy()
bad3["topk_weights"][0, 0, 0] = 1.5
try:
    C.assert_routing_valid(bad3)
    print("  FAIL: should have raised on weight=1.5"); sys.exit(1)
except C.ValueRangeError as e:
    print(f"  caught: {e}")
print("  all 3 negative cases caught")

print("=== test 3: attention value-range asserts ===")
n_layers, seq_len, n_heads = 6, 20, 96
good = {
    "per_head_entropy": np.random.uniform(0, np.log(seq_len), size=(n_layers, seq_len, n_heads)).astype(np.float32),
    "rowmax": np.random.uniform(0.1, 10, size=(n_layers, seq_len, n_heads)).astype(np.float32),
    "layers": np.array([12, 23, 39, 51, 62, 70], dtype=np.int32),
}
C.assert_attention_stats_valid(good, seq_len_upper_bound=seq_len)
print("  good payload: PASS")

bad = {"per_head_entropy": np.full((n_layers, seq_len, n_heads), -1.0, dtype=np.float32)}
try:
    C.assert_attention_stats_valid(bad)
    print("  FAIL: should have caught negative entropy"); sys.exit(1)
except C.ValueRangeError as e:
    print(f"  caught negative entropy: {e}")

bad = {"per_head_entropy": np.full((n_layers, seq_len, n_heads), 100.0, dtype=np.float32)}
try:
    C.assert_attention_stats_valid(bad, seq_len_upper_bound=seq_len)
    print("  FAIL: should have caught entropy>log(seq_len)"); sys.exit(1)
except C.ValueRangeError as e:
    print(f"  caught entropy>log(seq_len): {e}")

bad = {"per_head_entropy": np.full((n_layers, seq_len, n_heads), np.nan, dtype=np.float32)}
try:
    C.assert_attention_stats_valid(bad)
    print("  FAIL: should have caught NaN"); sys.exit(1)
except C.ValueRangeError as e:
    print(f"  caught NaN: {e}")

print("=== test 4: activations sanity ===")
good = {12: np.random.randn(20, 6144).astype(np.float32) * 10}
C.assert_activations_valid(good)
print("  good: PASS")
bad = {12: np.array([[np.inf]])}
try:
    C.assert_activations_valid(bad)
    print("  FAIL: should have caught inf"); sys.exit(1)
except C.ValueRangeError as e:
    print(f"  caught inf: {e}")

print("=== test 5: entropy non-degeneracy — degenerate hook ===")
# Synthetic degenerate case: same entropy every prompt
degen_results = []
for i in range(20):
    r = C.CaptureResult(prompt=f"p{i}", text="", raw={})
    r.attention_stats = {
        "per_head_entropy": np.full((6, 16, 96), 1.5, dtype=np.float32),  # CONSTANT
        "layers": np.array([12, 23, 39, 51, 62, 70], dtype=np.int32),
    }
    degen_results.append(r)
diag = C.compute_entropy_diagnostics(degen_results)
assert diag is not None
assert diag.entropy_std.max() < 1e-5, f"expected ~0 std, got {diag.entropy_std.max()}"
report = C.check_entropy_non_degenerate(diag, std_min=0.05, range_min=0.2)
assert report["status"] == "FAIL", report
assert report["failures"][0]["n_failing"] == diag.entropy_std.size
print(f"  PASS — caught {report['failures'][0]['n_failing']}/{diag.entropy_std.size} degenerate heads")

print("=== test 6: entropy non-degeneracy — varied hook ===")
varied_results = []
rng = np.random.default_rng(42)
for i in range(20):
    r = C.CaptureResult(prompt=f"p{i}", text="", raw={})
    # std ≈ 0.3 across prompts, range ≈ 1.0
    r.attention_stats = {
        "per_head_entropy": (1.5 + rng.normal(0, 0.3, size=(6, 16, 96))).astype(np.float32).clip(0, 5),
        "layers": np.array([12, 23, 39, 51, 62, 70], dtype=np.int32),
    }
    varied_results.append(r)
diag = C.compute_entropy_diagnostics(varied_results)
report_strict = C.check_entropy_non_degenerate(diag, std_min=0.05, range_min=0.2)
report_loose = C.check_entropy_non_degenerate(diag, std_min=None, range_min=None)
print(f"  varied: std summary: {report_strict['summary']}")
assert report_strict["status"] == "PASS", report_strict
assert report_loose["status"].startswith("report-only")
print("  PASS — both strict and report-only modes behave correctly")

print("=== test 7: residual NPZ save round-trip ===")
results = []
for i in range(5):
    r = C.CaptureResult(prompt=f"p{i}", text="", raw={})
    r.activations = {
        12: np.random.randn(8, 6144).astype(np.float32),
        62: np.random.randn(8, 6144).astype(np.float32),
    }
    results.append(r)
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "test.npz"
    C.save_residual_npz(p, results, [12, 62])
    loaded = np.load(p)
    assert "layer_012_last_tok" in loaded.files
    assert "layer_062_last_tok" in loaded.files
    assert loaded["layer_012_last_tok"].shape == (5, 6144)
print("  PASS")

print("=== test 8: quick_diagnostics produces expected pattern ===")
# Simulate a slight contrastive signal: toxic and benign vectors close at early layer, far at late
toxic_res, benign_res = [], []
for i in range(20):
    t = C.CaptureResult(prompt=f"t{i}", text="", raw={})
    b = C.CaptureResult(prompt=f"b{i}", text="", raw={})
    base = rng.normal(0, 1, size=6144).astype(np.float32)
    # Layer 12: nearly same direction; Layer 62: opposite directions
    t.activations = {12: (base + rng.normal(0, 0.1, size=(1, 6144))).astype(np.float32),
                     62: (base + rng.normal(0, 0.05, size=(1, 6144))).astype(np.float32)}
    b.activations = {12: (base + rng.normal(0, 0.1, size=(1, 6144))).astype(np.float32),
                     62: (-base + rng.normal(0, 0.05, size=(1, 6144))).astype(np.float32)}
    toxic_res.append(t); benign_res.append(b)
diag = C.quick_diagnostics(toxic_res, benign_res, [12, 62])
print(f"  layer 12 cos_sim={diag[12]['cos_similarity']:.3f} (expect high)")
print(f"  layer 62 cos_sim={diag[62]['cos_similarity']:.3f} (expect low/negative)")
assert diag[12]["cos_similarity"] > 0.9
assert diag[62]["cos_similarity"] < 0.0
print("  PASS")

def test_last_token_vecs_uses_prompt_tokens():
    # Synthesize a (1, seq_len=10, hidden=4) tensor with position-as-fingerprint
    n_layers, seq_len, hidden = 1, 10, 4
    fake = np.zeros((n_layers, seq_len, hidden), dtype=np.float32)
    for i in range(seq_len):
        fake[0, i, :] = i  # position i has value i
    r = C.CaptureResult(prompt="x", text="y", activations={12: fake[0]},
                        prompt_tokens=7)
    vecs = C._last_token_vecs([r], 12)
    assert vecs[0][0] == 6.0, f"expected position 6 (prompt_tokens-1), got {vecs[0][0]}"

print("\n*** all smoke tests passed ***")
