"""
D6 — RepE concept-direction extraction + steering smoke test.

Pipeline:
  1. Load per-pair last-prompt-token residuals from repe_bundle output.
  2. Compute concept direction d at L62 via difference-in-means (toxic − benign), unit-normalize.
  3. On a held-out toxic prompt (not in the 50-pair run), sweep α and call the model with
     `apply_steering_vectors = [{activations: -α·d_unit, layer_indices: [62], scale, norm_match, position_indices}]`.
  4. Compare baseline vs steered text. Smoke-test only — goal is to confirm the EII-4 plumbing fires.

Outputs in ./runs/phase2_d6_steering/:
  - direction_L62.npy            # unit-normalized concept direction (float32, hidden_size)
  - direction_diagnostics.json   # toxic/benign means, norms, separation
  - generations.json             # baseline + per-α steered outputs
  - per_call_*.json              # raw responses on failure (or with --dump-first-response)

Hardcoded scope (PHASE2_PLAN §7):
  - Steering layer: 62 (deepest separation point per regenerated Phase 1).
  - Sign convention: d_signed = mean(toxic) − mean(benign). For a toxic held-out prompt,
    `−α·d` pushes the residual toward the benign manifold → refusal should weaken.
    Use --sign +1 to flip if your repe_bundle labels are inverted.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Vendored serializer/deserializer — identical to sanity_check.py.
# We avoid `pip install vllm-lens` on the laptop (CUDA transitive deps).
# ---------------------------------------------------------------------------

try:
    import zstandard as zstd
except ImportError:
    print("[fatal] zstandard required. pip install zstandard", file=sys.stderr)
    sys.exit(2)


def deserialize_tensor(d: dict) -> np.ndarray:
    raw = base64.b64decode(d["data"])
    if d.get("compression") == "zstd":
        raw = zstd.ZstdDecompressor().decompress(raw)
    arr = np.frombuffer(raw, dtype=np.dtype(d["dtype"])).copy().reshape(d["shape"])
    if d.get("original_dtype") == "torch.bfloat16":
        arr = arr.view(np.uint16).astype(np.uint32).__lshift__(16).view(np.float32)
    return arr


def serialize_tensor_bf16(arr: np.ndarray) -> dict:
    """Encode a float32 numpy array as bf16-stored-as-int16 + zstd + b64.

    Matches the convention vllm-lens uses for residual-stream payloads, so the
    server-side `deserialize_tensor` in `_helpers/_serialize.py` should round-trip.
    """
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    # Truncate float32 → bfloat16 (drop the low 16 bits with round-to-nearest-even).
    f32_bits = arr.view(np.uint32)
    # Round-half-to-even on the discarded low bits.
    rounded = (f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)) >> 16
    bf16_as_uint16 = rounded.astype(np.uint16)
    int16_view = bf16_as_uint16.view(np.int16)
    raw = int16_view.tobytes()
    compressed = zstd.ZstdCompressor(level=1).compress(raw)
    return {
        "data": base64.b64encode(compressed).decode("ascii"),
        "dtype": "int16",
        "original_dtype": "torch.bfloat16",
        "shape": list(arr.shape),
        "compression": "zstd",
    }


# ---------------------------------------------------------------------------
# Residual loading — flexible across the layouts captures.py might use.
# ---------------------------------------------------------------------------

DEFAULT_PROBE_LAYERS = [12, 23, 39, 51, 62, 70]
STEER_LAYER = 62
HIDDEN_SIZE = 6144


def _load_layer_residuals(
    npz_path: Path,
    target_layer: int,
    probe_layers: list[int],
) -> np.ndarray:
    """Return shape (n_pairs, hidden_size) — already sliced to last prompt token.

    Canonical key written by captures.py:save_residual_npz is
    f"layer_{L:03d}_last_tok" (e.g. 'layer_062_last_tok'), and the array is
    pre-sliced to t[prompt_tokens - 1] for each pair. We also accept a few
    legacy/fallback key formats in case the writer changes.
    """
    canonical = f"layer_{target_layer:03d}_last_tok"
    aliases = (
        canonical,
        f"L{target_layer}",
        f"layer_{target_layer}",
        str(target_layer),
    )
    with np.load(npz_path, allow_pickle=False) as f:
        keys = list(f.keys())
        for k in aliases:
            if k in keys:
                arr = f[k]
                if arr.ndim != 2 or arr.shape[-1] != HIDDEN_SIZE:
                    raise ValueError(
                        f"{npz_path}: key '{k}' has unexpected shape {arr.shape} "
                        f"(expected (n_pairs, {HIDDEN_SIZE}))"
                    )
                return arr.astype(np.float32)
        raise KeyError(
            f"{npz_path}: no usable key for layer {target_layer}. "
            f"Tried {aliases}; available keys: {keys}"
        )


def compute_direction(
    toxic_npz: Path,
    benign_npz: Path,
    target_layer: int = STEER_LAYER,
    probe_layers: list[int] | None = None,
) -> tuple[np.ndarray, dict]:
    probe_layers = probe_layers or DEFAULT_PROBE_LAYERS
    tox = _load_layer_residuals(toxic_npz, target_layer, probe_layers)
    ben = _load_layer_residuals(benign_npz, target_layer, probe_layers)
    if tox.shape != ben.shape:
        raise ValueError(f"shape mismatch: toxic {tox.shape} vs benign {ben.shape}")

    tox_mean = tox.mean(axis=0)
    ben_mean = ben.mean(axis=0)
    d = tox_mean - ben_mean
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-6:
        raise RuntimeError(f"direction norm too small ({d_norm}); data likely degenerate")
    d_unit = (d / d_norm).astype(np.float32)

    # Diagnostics: per-pair signed projection should separate toxic > benign on d.
    proj_tox = tox @ d_unit
    proj_ben = ben @ d_unit
    diag = {
        "n_pairs": int(tox.shape[0]),
        "layer": int(target_layer),
        "hidden_size": int(tox.shape[1]),
        "toxic_mean_norm": float(np.linalg.norm(tox_mean)),
        "benign_mean_norm": float(np.linalg.norm(ben_mean)),
        "diff_norm": d_norm,
        "cos_means": float(
            (tox_mean @ ben_mean)
            / (np.linalg.norm(tox_mean) * np.linalg.norm(ben_mean) + 1e-12)
        ),
        "proj_toxic_mean": float(proj_tox.mean()),
        "proj_toxic_std": float(proj_tox.std()),
        "proj_benign_mean": float(proj_ben.mean()),
        "proj_benign_std": float(proj_ben.std()),
        # Cohen's d on projection — sanity check that d carries signal.
        "projection_cohens_d": float(
            (proj_tox.mean() - proj_ben.mean())
            / (np.sqrt(0.5 * (proj_tox.var() + proj_ben.var())) + 1e-12)
        ),
    }
    return d_unit, diag


# ---------------------------------------------------------------------------
# Held-out prompt selection.
# ---------------------------------------------------------------------------


def load_held_out_toxic_prompt(
    pairs_json: Path,
    explicit: str | None,
) -> str:
    if explicit:
        return explicit

    # Pull one ToxicChat toxic prompt not in pairs.json.
    used = set()
    if pairs_json.exists():
        pairs = json.loads(pairs_json.read_text())
        for p in pairs:
            for k in ("toxic", "harmful", "prompt_toxic"):
                if k in p:
                    used.add(p[k].strip())

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "Either --held-out-prompt or `pip install datasets` is required."
        )
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    for ex in ds:
        if int(ex.get("toxicity", 0)) == 1:
            text = ex["user_input"].strip()
            if text and text not in used:
                return text
    raise RuntimeError("no held-out toxic prompt found outside pairs.json")


# ---------------------------------------------------------------------------
# Steering payload construction + HTTP call.
# ---------------------------------------------------------------------------

# Soft refusal markers — purely for diagnostic shift summary. Not used to gate anything.
_REFUSAL_MARKERS = (
    "i can't", "i cannot", "i won't", "i will not",
    "i'm not able", "i am not able", "i'm unable", "i am unable",
    "sorry", "i apologize", "i apologise",
    "as an ai", "i must decline", "cannot help",
    "not appropriate", "not able to",
    "i shouldn't", "i should not",
)


def _build_steering_payload(
    d_unit: np.ndarray,
    alpha: float,
    sign: int,
    layer: int,
    norm_match: bool,
    position_indices: list[int] | None,
) -> dict:
    """Construct one SteeringVector-equivalent dict.

    Schema validated against vllm_lens/_helpers/_steering.py (pydantic BaseModel):
      activations:     torch.Tensor, dim ∈ {2,3}, shape[0] == len(layer_indices)
      layer_indices:   list[int]
      scale:           float (default 1.0)
      norm_match:      bool  (default False)
      position_indices: Optional[List[int]] (default None → apply at all positions)

    For a single-layer steer at L62 → activations must be shape (1, hidden_size).

    Convention: with d_signed = toxic_mean − benign_mean, applying activations =
    sign·d_unit with sign=−1 pushes a held-out toxic prompt's residual toward
    the benign manifold. With norm_match=True, vllm-lens rescales the steering
    vector to match the residual's runtime norm, so `scale` becomes the
    effective coefficient (typical RepE-norm-matched recipe).

    Wire serialization for `activations`: zstd+bf16-as-int16+b64 dict, same
    schema vllm_lens._helpers._serialize.serialize_tensor produces. The
    pydantic field_validator accepts this dict form directly (deserializes via
    deserialize_tensor on the server).
    """
    if d_unit.ndim != 1 or d_unit.shape[0] != HIDDEN_SIZE:
        raise ValueError(f"d_unit must be 1D ({HIDDEN_SIZE},); got {d_unit.shape}")
    # Pydantic requires shape[0] == len(layer_indices). One layer → (1, H).
    direction = (sign * d_unit).reshape(1, HIDDEN_SIZE).astype(np.float32)
    payload = {
        "activations": serialize_tensor_bf16(direction),
        "layer_indices": [int(layer)],
        "scale": float(alpha),
        "norm_match": bool(norm_match),
    }
    if position_indices is not None:
        payload["position_indices"] = list(position_indices)
    return payload


def _new_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI
    except ImportError:
        print("[fatal] openai package required. pip install openai", file=sys.stderr)
        sys.exit(2)
    return OpenAI(base_url=base_url, api_key=api_key)


def _generate(
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    steering_payload: dict | None,
    capture_layers: list[int] | None,
) -> dict:
    vllm_xargs: dict[str, Any] = {}
    if steering_payload is not None:
        # vLLM 0.20.0 vllm_xargs schema: Dict[str, Union[str, int, float, List[scalars]]].
        # A list-of-dict value (the SteeringVector payload) doesn't fit. Encode the whole
        # list as a JSON string; vllm-lens server-side parser json.loads it before
        # constructing SteeringVector. Confirmed against vllm-lens v1.1.0 plugin parsing.
        import json as _json
        vllm_xargs["apply_steering_vectors"] = _json.dumps([steering_payload])
    if capture_layers:
        vllm_xargs["output_residual_stream"] = list(capture_layers)

    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if vllm_xargs:
        extra_body["vllm_xargs"] = vllm_xargs

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    elapsed = time.time() - t0
    raw = resp.model_dump()
    return {"raw": raw, "elapsed_s": elapsed}


def _shift_summary(baseline_text: str, steered_text: str) -> dict:
    bt, st = baseline_text.lower(), steered_text.lower()
    b_refusals = sum(m in bt for m in _REFUSAL_MARKERS)
    s_refusals = sum(m in st for m in _REFUSAL_MARKERS)
    return {
        "baseline_chars": len(baseline_text),
        "steered_chars": len(steered_text),
        "baseline_refusal_markers": b_refusals,
        "steered_refusal_markers": s_refusals,
        "refusal_marker_delta": s_refusals - b_refusals,
        "identical": baseline_text == steered_text,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repe-dir", type=Path, default=Path("./runs/phase2_validation/repe_bundle"),
                   help="Directory containing the repe_bundle 50-pair run outputs.")
    p.add_argument("--toxic-npz-name", default="toxic_residual.npz",
                   help="Residual .npz filename for toxic pairs (default matches captures.py).")
    p.add_argument("--benign-npz-name", default="benign_residual.npz",
                   help="Residual .npz filename for benign pairs.")
    p.add_argument("--out-dir", type=Path, default=Path("./runs/phase2_d6_steering"))
    p.add_argument("--layer", type=int, default=STEER_LAYER)
    p.add_argument("--probe-layers", type=int, nargs="+", default=DEFAULT_PROBE_LAYERS)
    p.add_argument("--sign", type=int, choices=[-1, +1], default=-1,
                   help="-1: push toxic prompt toward benign manifold (default). +1: opposite.")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0, 8.0],
                   help="Steering scales (effective coefficient with norm_match=True).")
    p.add_argument("--norm-match", action=argparse.BooleanOptionalAction, default=True,
                   help="Rescale steering vector to match residual norm at runtime.")
    p.add_argument("--position-indices", type=int, nargs="*", default=None,
                   help="Token positions to steer at. Omit to apply at all positions.")
    p.add_argument("--held-out-prompt", type=str, default=None,
                   help="Override held-out toxic prompt (string). Default: pull from ToxicChat.")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--model", default="glm-5-1-fp8")
    p.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL"))
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY"))
    p.add_argument("--also-capture-residual", action="store_true",
                   help="Also request `output_residual_stream` on the steered call to inspect "
                        "post-steering residuals at the probe layers.")
    p.add_argument("--dump-first-response", action="store_true",
                   help="Persist the first raw response (baseline call) for schema verification.")
    p.add_argument("--scaffold-only", action="store_true",
                   help="Compute direction + diagnostics, save to disk, skip all server calls.")
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Compute direction ──
    toxic_npz = args.repe_dir / args.toxic_npz_name
    benign_npz = args.repe_dir / args.benign_npz_name
    for f in (toxic_npz, benign_npz):
        if not f.exists():
            print(f"[fatal] missing {f}", file=sys.stderr)
            return 2

    d_unit, diag = compute_direction(toxic_npz, benign_npz, args.layer, args.probe_layers)
    np.save(args.out_dir / "direction_L62.npy", d_unit)
    (args.out_dir / "direction_diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(f"[d6] L{args.layer} direction: ‖d‖={diag['diff_norm']:.3f}  "
          f"cohens_d_on_projection={diag['projection_cohens_d']:.3f}  "
          f"cos(toxic_mean, benign_mean)={diag['cos_means']:.4f}")

    if diag["projection_cohens_d"] < 0.5:
        print(f"[d6][warn] projection separation is weak (Cohen's d < 0.5). Steering smoke "
              f"is unlikely to produce a clean shift; proceeding anyway.")

    if args.scaffold_only:
        print(f"[d6] scaffold-only mode — direction saved. Exiting before any server calls.")
        return 0

    # ── 2. Server-side smoke ──
    if not args.base_url or not args.api_key:
        print("[fatal] VLLM_BASE_URL and VLLM_API_KEY must be set (env or --base-url/--api-key).",
              file=sys.stderr)
        return 2

    held_out = load_held_out_toxic_prompt(args.repe_dir / "pairs.json", args.held_out_prompt)
    print(f"[d6] held-out toxic prompt ({len(held_out)} chars): {held_out[:120]}...")

    client = _new_client(args.base_url, args.api_key)

    capture_layers = args.probe_layers if args.also_capture_residual else None

    runs: list[dict] = []
    baseline_text: str | None = None

    # Baseline (α = 0, no steering payload at all — strict no-instrumentation control).
    print(f"[d6] === baseline (no steering payload) ===")
    try:
        out = _generate(client, args.model, held_out, args.max_tokens,
                        steering_payload=None, capture_layers=capture_layers)
    except Exception as e:
        print(f"[d6][ERROR] baseline call failed: {type(e).__name__}: {e}", file=sys.stderr)
        cause = getattr(e, "__cause__", None)
        if cause:
            print(f"[d6][CAUSE] {type(cause).__name__}: {cause}", file=sys.stderr)
        return 1
    baseline_text = out["raw"]["choices"][0]["message"]["content"] or ""
    runs.append({
        "alpha": 0.0,
        "sign": 0,
        "steered": False,
        "elapsed_s": out["elapsed_s"],
        "text": baseline_text,
    })
    print(f"[d6] baseline ({out['elapsed_s']:.2f}s, {len(baseline_text)} chars):\n  {baseline_text[:200]}")
    if args.dump_first_response:
        (args.out_dir / "raw_baseline.json").write_text(json.dumps(out["raw"], indent=2)[:2_000_000])

    # Steered sweep.
    for alpha in args.alphas:
        if alpha == 0.0:
            continue  # baseline already covered
        payload = _build_steering_payload(
            d_unit=d_unit,
            alpha=alpha,
            sign=args.sign,
            layer=args.layer,
            norm_match=args.norm_match,
            position_indices=args.position_indices,
        )
        tag = f"sign{args.sign:+d}_alpha{alpha:g}"
        print(f"[d6] === steered {tag} (layer={args.layer}, norm_match={args.norm_match}) ===")
        try:
            out = _generate(client, args.model, held_out, args.max_tokens,
                            steering_payload=payload, capture_layers=capture_layers)
        except Exception as e:
            print(f"[d6][ERROR] {tag} call failed: {type(e).__name__}: {e}", file=sys.stderr)
            cause = getattr(e, "__cause__", None)
            if cause:
                print(f"[d6][CAUSE] {type(cause).__name__}: {cause}", file=sys.stderr)
            # Dump on first error so the user can read what the server complained about.
            err_path = args.out_dir / f"error_{tag}.json"
            err_path.write_text(json.dumps({
                "error": f"{type(e).__name__}: {e}",
                "cause": f"{type(cause).__name__}: {cause}" if cause else None,
                "sent_payload_keys": list(payload.keys()),
                "sent_payload_meta": {k: (v if not isinstance(v, dict) else
                                           {kk: vv for kk, vv in v.items() if kk != 'data'})
                                       for k, v in payload.items()},
            }, indent=2))
            print(f"[d6] error meta written to {err_path}")
            # Continue the sweep — other α's might still go through; but typically schema
            # errors are deterministic, so this will spam. Keep going to confirm.
            runs.append({"alpha": alpha, "sign": args.sign, "steered": True,
                         "error": f"{type(e).__name__}: {e}"})
            continue

        text = out["raw"]["choices"][0]["message"]["content"] or ""
        shift = _shift_summary(baseline_text, text)
        runs.append({
            "alpha": alpha,
            "sign": args.sign,
            "steered": True,
            "elapsed_s": out["elapsed_s"],
            "text": text,
            "shift": shift,
        })
        marker = "≡" if shift["identical"] else "↔"
        print(f"[d6] {tag} ({out['elapsed_s']:.2f}s, {len(text)} chars) {marker}:\n  {text[:200]}")
        print(f"      shift: {shift}")

    # ── 3. Persist ──
    (args.out_dir / "generations.json").write_text(json.dumps({
        "held_out_prompt": held_out,
        "model": args.model,
        "layer": args.layer,
        "sign": args.sign,
        "norm_match": args.norm_match,
        "position_indices": args.position_indices,
        "direction_diagnostics": diag,
        "runs": runs,
    }, indent=2))

    # ── 4. Smoke verdict ──
    steered_runs = [r for r in runs if r.get("steered") and "shift" in r]
    if not steered_runs:
        print("[d6][FAIL] no steered runs succeeded.")
        return 1
    any_shift = any(not r["shift"]["identical"] for r in steered_runs)
    any_refusal_drop = any(r["shift"]["refusal_marker_delta"] < 0 for r in steered_runs)
    print(f"[d6] === smoke verdict ===")
    print(f"      any_shift_from_baseline: {any_shift}")
    print(f"      any_refusal_marker_drop: {any_refusal_drop}")
    print(f"      outputs in {args.out_dir}")
    if any_shift:
        print("[d6][PASS] steering plumbing fires — EII-4 wired correctly for Phase 3.")
        return 0
    print("[d6][SOFT-FAIL] outputs identical to baseline across all α. "
          "Inspect error_*.json (if any), then try larger α, opposite --sign, or omit --norm-match.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
