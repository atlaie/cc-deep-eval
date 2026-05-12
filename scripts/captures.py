"""
captures.py — Phase 1 + Phase 2 client-side capture library.

Self-contained extraction of the shared client logic. Replaces the inline
functions in sanity_check.py. sanity_check.py is preserved unchanged until
parity is confirmed; once verified, it becomes a thin wrapper around this
module.

No CUDA-bound deps (macOS arm64 safe). Only numpy, zstandard, datasets, openai.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zstandard as zstd
from datasets import load_dataset
from openai import OpenAI


# ===== GLM 5.1 architectural constants ======================================

GLM51_NUM_LAYERS = 78
GLM51_FIRST_DENSE_LAYERS = 3  # layers 0..2 dense, 3..77 MoE
GLM51_HIDDEN_SIZE = 6144
GLM51_N_ROUTED_EXPERTS = 256
GLM51_MOE_TOP_K = 8

# Capture defaults
DEFAULT_PROBE_LAYERS = [12, 23, 39, 51, 62, 70]
DEFAULT_ROUTING_LAYERS = list(range(GLM51_FIRST_DENSE_LAYERS, GLM51_NUM_LAYERS))  # all 75 MoE


# ===== data containers ======================================================

@dataclass
class Pair:
    pair_id: int
    toxic: str
    benign: str


@dataclass
class CaptureRequest:
    """What this call should ask the server to capture.

    A value of None means 'do not request that payload kind'. An empty list
    means 'request the payload kind with no layer filter' (server default
    applies). xargs are orthogonal; multiple may be set simultaneously.
    """
    residual_layers: list[int] | None = None
    routing_layers: list[int] | None = None
    attention_layers: list[int] | None = None

    def to_xargs(self) -> dict[str, Any]:
        x: dict[str, Any] = {}
        if self.residual_layers is not None:
            x["output_residual_stream"] = self.residual_layers
        if self.routing_layers is not None:
            x["output_routing"] = self.routing_layers
        if self.attention_layers is not None:
            x["output_attention_stats"] = self.attention_layers
        return x

    def any_capture(self) -> bool:
        return any(v is not None for v in (
            self.residual_layers, self.routing_layers, self.attention_layers,
        ))


@dataclass
class CaptureResult:
    """Single-call result. Numpy arrays held in memory; serialize via helpers."""
    prompt: str
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    activations: dict[int, np.ndarray] | None = None
    routing: dict[str, np.ndarray] | None = None
    attention_stats: dict[str, np.ndarray] | None = None
    wall_seconds: float = 0.0
    payload_bytes: int = 0          # rough on-wire size estimate
    prompt_tokens: int = 0
    error: str | None = None

    def to_meta_json(self) -> dict[str, Any]:
        """JSON-safe metadata (no large arrays)."""
        return {
            "prompt": self.prompt,
            "text": self.text,
            "wall_seconds": self.wall_seconds,
            "payload_bytes": self.payload_bytes,
            "error": self.error,
            "has_activations": self.activations is not None,
            "has_routing": self.routing is not None,
            "has_attention_stats": self.attention_stats is not None,
        }


# ===== deserialization ======================================================

def deserialize_tensor(d: dict[str, Any]) -> np.ndarray:
    """Canonical numpy-only deserializer (PHASE1_REFERENCE §6.4).

    Schema keys:
      data:           base64 zstd-compressed raw bytes
      dtype:          on-wire dtype ('int16' for bf16-as-int16 OR native int16)
      original_dtype: 'torch.bfloat16' → reinterpret int16→fp32
                      'torch.int16'    → leave as int16 indices
      shape:          target shape
      compression:    'zstd' or absent

    Mismatched original_dtype is the canonical silent-corruption bug (treats
    int16 expert indices as bf16 bits and returns garbage floats).
    """
    raw = base64.b64decode(d["data"])
    if d.get("compression") == "zstd":
        raw = zstd.ZstdDecompressor().decompress(raw)
    arr = np.frombuffer(raw, dtype=np.dtype(d["dtype"])).copy().reshape(d["shape"])
    if d.get("original_dtype") == "torch.bfloat16":
        # bf16 bits stored as int16 → zero-pad upper 16 bits → reinterpret as fp32
        arr = arr.view(np.uint16).astype(np.uint32).__lshift__(16).view(np.float32)
    return arr


# ===== contrastive pair construction ========================================

def build_contrastive_pairs(
    n_pairs: int,
    max_prompt_chars: int = 600,
    seed: int = 0,
) -> list[Pair]:
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    toxic = ds.filter(lambda r: r["toxicity"] == 1 and len(r["user_input"]) < max_prompt_chars)
    benign = ds.filter(lambda r: r["toxicity"] == 0 and len(r["user_input"]) < max_prompt_chars)
    if len(toxic) < n_pairs or len(benign) < n_pairs:
        raise RuntimeError(
            f"Not enough samples after filter: toxic={len(toxic)} benign={len(benign)} "
            f"requested={n_pairs}. Loosen --max-prompt-chars or reduce --n-pairs."
        )
    toxic = toxic.shuffle(seed=seed).select(range(n_pairs))
    benign = benign.shuffle(seed=seed).select(range(n_pairs))
    return [
        Pair(pair_id=i, toxic=toxic[i]["user_input"], benign=benign[i]["user_input"])
        for i in range(n_pairs)
    ]


# ===== single API call ======================================================

def call_with_capture(
    client: OpenAI,
    model: str,
    user_prompt: str,
    request: CaptureRequest,
    max_new_tokens: int,
    dump_path: Path | None = None,
) -> CaptureResult:
    """One chat-completion call. Captures whatever payloads `request` toggles."""
    t0 = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=max_new_tokens,
            extra_body={
                "vllm_xargs": request.to_xargs(),
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    except Exception as e:
        return CaptureResult(
            prompt=user_prompt, text="", raw={},
            wall_seconds=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}",
        )
    wall = time.perf_counter() - t0
    raw = response.model_dump()

    if dump_path is not None:
        dump_path.write_text(json.dumps(raw, indent=2, default=str))

    text = response.choices[0].message.content or ""
    usage = raw.get("usage") or {}
    res = CaptureResult(
        prompt=user_prompt, text=text, raw=raw,
        wall_seconds=wall, payload_bytes=_estimate_payload_bytes(raw),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
    )
    # Only attempt extraction for what was requested — keeps errors local.
    if request.residual_layers is not None:
        res.activations = extract_activations(raw, request.residual_layers)
    if request.routing_layers is not None:
        res.routing = extract_routing(raw)
    if request.attention_layers is not None:
        res.attention_stats = extract_attention_stats(raw)
    return res


def _estimate_payload_bytes(raw: dict) -> int:
    """Rough on-wire size: sum of base64 string lengths of every leaf 'data' key."""
    total = 0

    def _walk(node):
        nonlocal total
        if isinstance(node, dict):
            if "data" in node and isinstance(node["data"], str):
                total += len(node["data"])
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    for key in ("activations", "routing", "attention_stats", "gradients"):
        if key in raw:
            _walk(raw[key])
    return total


# ===== payload extractors ===================================================

def extract_activations(raw: dict, layers: list[int]) -> dict[int, np.ndarray] | None:
    """Residual stream → {layer_idx: array}. Matches sanity_check.py behavior."""
    blob = raw.get("activations")
    if blob is None:
        return None
    rs = blob.get("residual_stream")
    if rs is None:
        return None
    try:
        arr = np.asarray(deserialize_tensor(rs))  # (n_layers, seq_len, hidden_size)
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        return {layer_idx: arr[i] for i, layer_idx in enumerate(layers) if i < arr.shape[0]}
    except Exception as e:
        print(f"[warn] residual_stream deserialize failed: {e}")
        return None


def extract_routing(raw: dict) -> dict[str, np.ndarray] | None:
    """MoE routing payload (PHASE2_PLAN §4.1).

    Returns a dict with whichever subset of these keys arrives:
      topk_ids       int16,   (n_layers, seq_len, top_k)
      topk_weights   fp32,    (n_layers, seq_len, top_k)
      routing_entropy fp32,   (n_layers, seq_len)
      layer_indices  int32,   (n_layers,)
    """
    blob = raw.get("routing")
    if not isinstance(blob, dict):
        return None
    out: dict[str, np.ndarray] = {}
    try:
        for key in ("topk_ids", "topk_weights", "routing_entropy"):
            if key in blob and isinstance(blob[key], dict):
                out[key] = np.asarray(deserialize_tensor(blob[key]))
        if "layer_indices" in blob:
            out["layer_indices"] = np.asarray(blob["layer_indices"], dtype=np.int32)
    except Exception as e:
        print(f"[warn] routing deserialize failed: {e}")
        return None
    return out or None


def extract_attention_stats(raw: dict) -> dict[str, np.ndarray] | None:
    """Attention summary stats (PHASE2_PLAN §4.2).

    Server-side shape (verified on hardware 2026-05-12):
      (n_layers, n_heads, seq_len)  — note: seq_len is the LAST axis, not the middle.

    Canonical keys stored in the returned dict (normalised from server names):
      per_head_entropy    fp32, (n_layers, n_heads, seq_len)
      rowmax              fp32, (n_layers, n_heads, seq_len)  [server: per_head_max]
      top_mass            fp32, (n_layers, n_heads, seq_len)  [server: top10pct_mass]
      layer_indices       int32,(n_layers,)                   [server: layer_indices]

    Server-side names may diverge from Plan §4.2 names; this function normalises them
    so the rest of the pipeline uses stable keys.
    """
    blob = raw.get("attention_stats")
    if not isinstance(blob, dict):
        return None
    # Canonical name → list of server-side aliases to check (first match wins)
    KEY_MAP = {
        "per_head_entropy": ["per_head_entropy"],
        "rowmax":           ["per_head_max", "rowmax"],
        "top_mass":         ["top10pct_mass", "topk_attended_positions"],
    }
    out: dict[str, np.ndarray] = {}
    try:
        for canonical, aliases in KEY_MAP.items():
            for alias in aliases:
                if alias in blob and isinstance(blob[alias], dict):
                    out[canonical] = np.asarray(deserialize_tensor(blob[alias]))
                    break
        for layer_key in ("layer_indices", "layers"):
            if layer_key in blob:
                out["layer_indices"] = np.asarray(blob[layer_key], dtype=np.int32)
                break
    except Exception as e:
        print(f"[warn] attention_stats deserialize failed: {e}")
        return None
    return out or None


# ===== value-range assertions (EXTENSION_PHASE2 §11) =========================

class ValueRangeError(AssertionError):
    """Raised when a payload's values are outside their architectural range.

    A failure here means the hook is reading the wrong tensor (silent
    corruption) — NOT a soft warning. Phase 2's routing extension hit this
    exact failure on the first build; that's why this check is mandatory.
    """


def assert_routing_valid(
    routing: dict[str, np.ndarray],
    n_experts: int = GLM51_N_ROUTED_EXPERTS,
    top_k: int = GLM51_MOE_TOP_K,
) -> None:
    ids = routing.get("topk_ids")
    wts = routing.get("topk_weights")
    if ids is None:
        raise ValueRangeError("routing payload missing topk_ids")
    if ids.shape[-1] != top_k:
        raise ValueRangeError(f"topk_ids last dim {ids.shape[-1]} != top_k {top_k}")
    if int(ids.max()) >= n_experts:
        raise ValueRangeError(f"topk_ids.max()={int(ids.max())} >= n_experts={n_experts}")
    if int(ids.min()) < 0:
        raise ValueRangeError(f"topk_ids.min()={int(ids.min())} < 0")
    if wts is not None:
        if not np.isfinite(wts).all():
            raise ValueRangeError("topk_weights contains NaN/inf")
        if (wts < 0).any() or (wts > 1).any():
            raise ValueRangeError(
                f"topk_weights outside sigmoid range [0,1]: "
                f"min={float(wts.min()):.4f} max={float(wts.max()):.4f}"
            )


def assert_attention_stats_valid(
    stats: dict[str, np.ndarray],
    seq_len_upper_bound: int | None = None,
) -> None:
    """Range-only checks. Non-degeneracy is corpus-level (see entropy diagnostics).

    Shape convention (verified on hardware): (n_layers, n_heads, seq_len).
    seq_len_upper_bound applies to the last axis.
    """
    ent = stats.get("per_head_entropy")
    if ent is None:
        raise ValueRangeError("attention_stats missing per_head_entropy")
    if not np.isfinite(ent).all():
        raise ValueRangeError("per_head_entropy contains NaN/inf")
    if (ent < 0).any():
        raise ValueRangeError(f"per_head_entropy has negatives; min={float(ent.min()):.4f}")
    if seq_len_upper_bound is not None and seq_len_upper_bound > 1:
        # seq_len is axis 2; entropy upper bound is log(seq_len)
        max_allowed = float(np.log(seq_len_upper_bound)) * 1.05 + 1e-3
        if float(ent.max()) > max_allowed:
            raise ValueRangeError(
                f"per_head_entropy.max()={float(ent.max()):.4f} exceeds "
                f"1.05·log({seq_len_upper_bound})={max_allowed:.4f}"
            )
    rm = stats.get("rowmax")  # canonical name after extract_attention_stats normalises
    if rm is not None:
        if not np.isfinite(rm).all():
            raise ValueRangeError("rowmax (per_head_max) contains NaN/inf")
        if (rm <= 0).any():
            raise ValueRangeError(f"rowmax (per_head_max) non-positive; min={float(rm.min()):.4f}")


def assert_activations_valid(activations: dict[int, np.ndarray]) -> None:
    """Sanity checks for residual stream — magnitude/finite only."""
    for L, arr in activations.items():
        if not np.isfinite(arr).all():
            raise ValueRangeError(f"layer {L} activations contain NaN/inf")
        m = float(np.abs(arr).max())
        if m > 2000:
            # GLM 5.1 residual stream typically peaks in low-hundreds; >2000 = garbage
            raise ValueRangeError(f"layer {L} |activations|.max()={m:.1f} > 2000 sanity bound")


# ===== diagnostics — residual stream (Phase 1) ==============================

def _last_token_vecs(results: list[CaptureResult], layer: int) -> list[np.ndarray]:
    out = []
    for r in results:
        if r.activations is None or layer not in r.activations:
            continue
        t = r.activations[layer]
        if t.ndim == 3:
            t = t[0]
        # seq_len axis spans prefill (prompt_tokens) + (completion_tokens - 1) decode steps.
        # We want the last *prompt* token's residual: index prompt_tokens - 1.
        if r.prompt_tokens <= 0 or r.prompt_tokens > t.shape[0]:
            # Schema drift or missing usage; loud-warn and fall back to t[-1] so we
            # don't silently produce garbage.
            print(f"[warn] layer {layer}: prompt_tokens={r.prompt_tokens} vs "
                  f"seq_len={t.shape[0]}; falling back to t[-1]")
            out.append(t[-1])
        else:
            out.append(t[r.prompt_tokens - 1])
    return out


def quick_diagnostics(
    toxic: list[CaptureResult],
    benign: list[CaptureResult],
    layers: list[int],
) -> dict[int, dict[str, float]]:
    """Per-layer difference-in-means cos-sim on last-token residual stream.

    For GLM 5.1 (hidden=6144), expected pattern:
      Layer 12 (early): cos ~0.97-0.99
      Layer 39 (mid):   cos ~0.90-0.95
      Layer 62 (late):  cos <0.88
    Flat across all layers = capture broken or data not contrastive.
    """
    summary: dict[int, dict[str, float]] = {}
    for L in layers:
        tv = _last_token_vecs(toxic, L)
        bv = _last_token_vecs(benign, L)
        if not tv or not bv:
            continue
        tm = np.stack(tv).mean(axis=0)
        bm = np.stack(bv).mean(axis=0)
        cos = float(np.dot(tm, bm) / (np.linalg.norm(tm) * np.linalg.norm(bm) + 1e-8))
        summary[L] = {
            "n_toxic": len(tv),
            "n_benign": len(bv),
            "diff_norm": float(np.linalg.norm(tm - bm)),
            "toxic_norm": float(np.linalg.norm(tm)),
            "benign_norm": float(np.linalg.norm(bm)),
            "cos_similarity": cos,
        }
    return summary


# ===== diagnostics — attention entropy non-degeneracy =======================

@dataclass
class EntropyDiagnostics:
    """Per-(layer, head) entropy distribution statistics across the corpus.

    All arrays are shape (n_layers_in_capture_set, n_heads).
    """
    layers: np.ndarray
    entropy_mean: np.ndarray
    entropy_std: np.ndarray
    entropy_min: np.ndarray
    entropy_max: np.ndarray
    n_prompts: int
    aggregation: str  # 'last_token' | 'mean_over_seq'

    def to_json(self) -> dict[str, Any]:
        return {
            "layers": self.layers.tolist(),
            "n_prompts": self.n_prompts,
            "aggregation": self.aggregation,
            "entropy_mean": self.entropy_mean.tolist(),
            "entropy_std": self.entropy_std.tolist(),
            "entropy_min": self.entropy_min.tolist(),
            "entropy_max": self.entropy_max.tolist(),
        }


def compute_entropy_diagnostics(
    results: list[CaptureResult],
    aggregation: str = "last_token",
) -> EntropyDiagnostics | None:
    """Aggregate per-head entropy across the corpus.

    Server shape (verified on hardware): (n_layers, n_heads, seq_len).
    seq_len varies per prompt (last axis); n_heads is fixed (axis 1).

    aggregation:
      'last_token'    → use entropy at the final query position (ent[:, :, -1])
      'mean_over_seq' → mean across all query positions (ent.mean(axis=2))

    Both collapse the seq_len axis, yielding (n_layers, n_heads) per prompt.
    Std across prompts is then computed over (n_prompts, n_layers, n_heads).
    """
    sample = next(
        (r for r in results if r.attention_stats and "per_head_entropy" in r.attention_stats),
        None,
    )
    if sample is None:
        return None
    ent_sample = sample.attention_stats["per_head_entropy"]
    if ent_sample.ndim != 3:
        return None
    # Shape: (n_layers, n_heads, seq_len) — seq_len varies, so filter on first two dims only.
    n_layers, n_heads, _ = ent_sample.shape

    layer_indices = sample.attention_stats.get("layer_indices")
    if layer_indices is None:
        layer_indices = np.arange(n_layers, dtype=np.int32)

    per_prompt: list[np.ndarray] = []
    for r in results:
        if r.attention_stats is None:
            continue
        ent = r.attention_stats.get("per_head_entropy")
        if ent is None or ent.ndim != 3:
            continue
        if ent.shape[0] != n_layers or ent.shape[1] != n_heads:
            # Different n_layers or n_heads → skip (seq_len mismatch on axis 2 is expected)
            continue
        if aggregation == "last_token":
            per_prompt.append(ent[:, :, r.prompt_tokens - 1])  #(n_layers, n_heads); was ent[:, :, -1] before
        elif aggregation == "mean_over_seq":
            per_prompt.append(ent.mean(axis=2))     # (n_layers, n_heads)
        else:
            raise ValueError(f"unknown aggregation: {aggregation}")
    if not per_prompt:
        return None
    stacked = np.stack(per_prompt, axis=0)  # (n_prompts, n_layers, n_heads)
    return EntropyDiagnostics(
        layers=np.asarray(layer_indices),
        entropy_mean=stacked.mean(axis=0),
        entropy_std=stacked.std(axis=0),
        entropy_min=stacked.min(axis=0),
        entropy_max=stacked.max(axis=0),
        n_prompts=stacked.shape[0],
        aggregation=aggregation,
    )


def check_entropy_non_degenerate(
    diag: EntropyDiagnostics,
    std_min: float | None,
    range_min: float | None,
) -> dict[str, Any]:
    """Apply non-degeneracy criteria.

    std_min:   per-(layer,head) std across prompts must exceed this.
               Rejects 'all near zero', 'all near log(seq_len)', constants.
    range_min: per-(layer,head) max-min across prompts must exceed this.
               Rejects broadcasts that happen to land mid-range.

    Either argument may be None → report only, no assertion. This lets the
    caller eyeball the distribution from a smoke run before fixing thresholds.

    Returns a report dict; status is 'report-only' (no thresholds set),
    'PASS', or 'FAIL'.
    """
    rng = diag.entropy_max - diag.entropy_min
    report: dict[str, Any] = {
        "thresholds": {"std_min": std_min, "range_min": range_min},
        "n_prompts": diag.n_prompts,
        "aggregation": diag.aggregation,
        "summary": {
            "min_std": float(diag.entropy_std.min()),
            "median_std": float(np.median(diag.entropy_std)),
            "mean_std": float(diag.entropy_std.mean()),
            "max_std": float(diag.entropy_std.max()),
            "min_range": float(rng.min()),
            "median_range": float(np.median(rng)),
            "max_range": float(rng.max()),
        },
    }
    if std_min is None and range_min is None:
        report["status"] = "report-only (no thresholds set)"
        return report

    failures: list[dict[str, Any]] = []
    if std_min is not None:
        bad = np.argwhere(diag.entropy_std <= std_min)
        if bad.size:
            failures.append({
                "criterion": f"entropy_std > {std_min}",
                "n_failing": int(bad.shape[0]),
                "total_heads": int(diag.entropy_std.size),
                "examples": [
                    {"layer": int(diag.layers[i]), "head": int(h)}
                    for i, h in bad[:5].tolist()
                ],
            })
    if range_min is not None:
        bad = np.argwhere(rng <= range_min)
        if bad.size:
            failures.append({
                "criterion": f"entropy_max - entropy_min > {range_min}",
                "n_failing": int(bad.shape[0]),
                "total_heads": int(rng.size),
                "examples": [
                    {"layer": int(diag.layers[i]), "head": int(h)}
                    for i, h in bad[:5].tolist()
                ],
            })
    report["failures"] = failures
    report["status"] = "PASS" if not failures else "FAIL"
    return report


# ===== saving ===============================================================

def save_residual_npz(path: Path, results: list[CaptureResult], layers: list[int]) -> None:
    """Last-token residual per layer → .npz, one (n_pairs, hidden) array per layer."""
    bundles: dict[str, np.ndarray] = {}
    for L in layers:
        vecs = _last_token_vecs(results, L)
        if vecs:
            bundles[f"layer_{L:03d}_last_tok"] = np.stack(vecs)
    if bundles:
        np.savez(path, **bundles)


def _make_ragged(arrs: list[np.ndarray]) -> np.ndarray:
    """Safely create a 1-D object array from a list of arrays with varying shapes.

    np.array(arrs, dtype=object) fails when shapes differ in non-first dimensions
    (numpy tries to broadcast and errors). np.empty + element assignment bypasses
    that inference and always produces a correct 1-D object array.
    """
    out = np.empty(len(arrs), dtype=object)
    for i, a in enumerate(arrs):
        out[i] = a
    return out


def save_routing_npz(path: Path, results: list[CaptureResult]) -> None:
    """Routing arrays per prompt → .npz. Variable seq_len handled via object dtype."""
    has = [r for r in results if r.routing is not None]
    if not has:
        return
    sample = has[0].routing
    bundles: dict[str, np.ndarray] = {}
    for key in ("topk_ids", "topk_weights", "routing_entropy"):
        arrs = [r.routing[key] for r in has if key in r.routing]
        if not arrs:
            continue
        try:
            bundles[key] = np.stack(arrs)
        except ValueError:
            bundles[key] = _make_ragged(arrs)
    if "layer_indices" in sample:
        bundles["layer_indices"] = sample["layer_indices"]
    if bundles:
        np.savez(path, **bundles, allow_pickle=True)


def save_attention_npz(path: Path, results: list[CaptureResult]) -> None:
    """Attention stats arrays per prompt → .npz."""
    has = [r for r in results if r.attention_stats is not None]
    if not has:
        return
    sample = has[0].attention_stats
    bundles: dict[str, np.ndarray] = {}
    for key in ("per_head_entropy", "rowmax", "topk_attended_positions"):
        arrs = [r.attention_stats[key] for r in has if key in r.attention_stats]
        if not arrs:
            continue
        try:
            bundles[key] = np.stack(arrs)
        except ValueError:
            bundles[key] = _make_ragged(arrs)
    if "layers" in sample:
        bundles["layers"] = sample["layers"]
    if bundles:
        np.savez(path, **bundles, allow_pickle=True)