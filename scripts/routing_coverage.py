#!/usr/bin/env python3
"""
routing_coverage.py — Phase 2 D5 routing coverage check.

Validates that captured `topk_ids` cover > 50% of the 256-expert space per MoE layer
across the corpus, per PHASE2_PLAN §7 / PHASE2_REFERENCE.md §10. Operates as a
post-aggregate step on the artifacts already on disk in
`runs/phase2_validation/routing/`; does not re-issue any HTTP calls.

Standalone — deliberately does not import `captures.py` (avoids the two-diverging-copies
trap from PHASE2_PLAN §2.5). Inlines the minimal numpy-only deserializer.

Loading strategies, tried in order:
  1. Per-class NPZ files (`toxic*routing*.npz`, `benign*routing*.npz`)
  2. Any other NPZ in the run dir
     - Sub-layout A: 'topk_ids' (N-D) + 'layer_indices' (1-D)
     - Sub-layout B: per-layer keys 'topk_ids_L<idx>' or 'L<idx>'
  3. Raw response JSON files with a top-level 'routing' field (fallback)

Exit codes:
  0 — pass (all MoE layers ≥ threshold)
  2 — user error (run-dir missing, no recognizable routing data)
  3 — coverage check failed (≥ 1 MoE layer below threshold)
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import zstandard as zstd
except ImportError:
    zstd = None

N_ROUTED_EXPERTS = 256
COVERAGE_THRESHOLD = 0.50
FIRST_MOE_LAYER = 3       # GLM 5.1: first_k_dense_replace = 3
N_MOE_LAYERS = 75         # GLM 5.1: 75 MoE layers, indices 3..77


# ---------------------------------------------------------------------------
# Serialization (mirrors captures.py / sanity_check.py canonical form)
# ---------------------------------------------------------------------------

def deserialize_tensor(d: dict) -> np.ndarray:
    """Decode a zstd-compressed base64 tensor blob into a numpy array.

    Mirrors the canonical Phase 1/2 client-side deserializer. For int16 index
    tensors (`topk_ids`), `original_dtype == 'torch.int16'` and no bf16 reinterpret
    is performed.
    """
    raw = base64.b64decode(d["data"])
    if d.get("compression") == "zstd":
        if zstd is None:
            raise RuntimeError(
                "zstandard package required to decode zstd-compressed payloads "
                "(pip install zstandard)"
            )
        raw = zstd.ZstdDecompressor().decompress(raw)
    arr = np.frombuffer(raw, dtype=np.dtype(d["dtype"])).copy().reshape(d["shape"])
    if d.get("original_dtype") == "torch.bfloat16":
        arr = arr.view(np.uint16).astype(np.uint32).__lshift__(16).view(np.float32)
    return arr


# ---------------------------------------------------------------------------
# Loaders (each returns {layer_idx: [arr, ...]} or empty dict on no-match)
# ---------------------------------------------------------------------------

def _load_npz_stacked(npz_path: Path, exclude_bos: bool = False) -> dict[int, list[np.ndarray]]:
    """NPZ layout A: `topk_ids` + 1-D `layer_indices`.

    Two sub-cases for `topk_ids`:
      * regular N-D ndarray (e.g. all prompts had identical seq_len → np.stack
        succeeded): auto-detect the layer axis by matching `len(layer_indices)`.
        `exclude_bos` is ignored here (token axis not reliably identifiable from
        shape alone — would need a top_k=8 axis hint).
      * 1-D object array of length n_prompts, each element a `(n_layers,
        seq_len_i, top_k)` ndarray: the canonical output of
        `captures.save_routing_npz` via `_make_ragged`. Token axis is axis 1 of
        each element; `exclude_bos` drops it before bucketing.
    """
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            if "topk_ids" not in d.files or "layer_indices" not in d.files:
                return {}
            stack = np.asarray(d["topk_ids"])
            layer_indices = np.asarray(d["layer_indices"]).tolist()
    except (OSError, ValueError, EOFError):
        return {}

    n_layers = len(layer_indices)

    # Sub-case: 1-D object array of per-prompt ndarrays (captures.py ragged path)
    if stack.dtype == object:
        per_layer: dict[int, list[np.ndarray]] = {int(li): [] for li in layer_indices}
        for elem in stack.ravel():
            if elem is None:
                continue
            arr = np.asarray(elem)
            if arr.ndim < 2 or arr.shape[0] != n_layers:
                continue
            for i, li in enumerate(layer_indices):
                per_token = arr[i]  # (seq_len, top_k)
                if exclude_bos and per_token.shape[0] >= 2:
                    per_token = per_token[1:]
                per_layer[int(li)].append(per_token)
        return {li: arrs for li, arrs in per_layer.items() if arrs}

    # Sub-case: regular N-D ndarray. Find a layer axis by length match.
    if exclude_bos:
        print(f"  warn: --exclude-bos requested but {npz_path.name} stores a regular "
              f"N-D ndarray; token axis cannot be reliably inferred from shape — "
              f"BOS exclusion NOT applied to this file.", file=sys.stderr)
    axes = [i for i, s in enumerate(stack.shape) if s == n_layers]
    if not axes:
        return {}
    layer_axis = axes[0]
    stack = np.moveaxis(stack, layer_axis, 0)
    return {int(li): [stack[i]] for i, li in enumerate(layer_indices)}


_KEY_RE = re.compile(r"^(?:topk_ids_)?L(\d+)$")


def _load_npz_keyed_by_layer(npz_path: Path, exclude_bos: bool = False) -> dict[int, list[np.ndarray]]:
    """NPZ layout B: per-layer keys like 'L3', 'L4', ... or 'topk_ids_L3', ..."""
    out: dict[int, list[np.ndarray]] = {}
    try:
        with np.load(npz_path, allow_pickle=True) as d:
            for k in d.files:
                m = _KEY_RE.match(k)
                if m:
                    out[int(m.group(1))] = [d[k]]
    except (OSError, ValueError, EOFError):
        return {}
    if exclude_bos and out:
        print(f"  warn: --exclude-bos requested but {npz_path.name} uses per-layer-keyed "
              f"layout; token axis cannot be reliably inferred — exclusion NOT applied.",
              file=sys.stderr)
    return out


def _load_response_json(json_path: Path, exclude_bos: bool = False) -> dict[int, list[np.ndarray]]:
    """Single raw response JSON with a top-level 'routing' field."""
    try:
        with open(json_path) as f:
            resp = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(resp, dict):
        return {}
    routing = resp.get("routing")
    if not isinstance(routing, dict):
        return {}
    payload = routing.get("topk_ids")
    layer_indices = routing.get("layer_indices")
    if not isinstance(payload, dict) or not isinstance(layer_indices, list):
        return {}
    try:
        arr = deserialize_tensor(payload)
    except Exception as exc:
        print(f"  warn: failed to deserialize {json_path.name}: {exc}", file=sys.stderr)
        return {}
    if arr.ndim < 1 or arr.shape[0] != len(layer_indices):
        return {}
    out: dict[int, list[np.ndarray]] = {}
    for i, li in enumerate(layer_indices):
        per_token = arr[i]  # (seq_len, top_k) in the documented schema
        if exclude_bos and per_token.ndim >= 1 and per_token.shape[0] >= 2:
            per_token = per_token[1:]
        out[int(li)] = [per_token]
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _merge(dest: dict[int, list[np.ndarray]], src: dict[int, list[np.ndarray]]) -> None:
    for li, arrs in src.items():
        dest.setdefault(li, []).extend(arrs)


def collect_class(run_dir: Path, class_filter: str | None,
                  exclude_bos: bool = False) -> tuple[dict[int, list[np.ndarray]], list[str]]:
    """Collect topk_ids per layer for one class (or all data if class_filter is None).

    Returns (per_layer_data, list_of_source_descriptions).
    """
    per_layer: dict[int, list[np.ndarray]] = {}
    sources: list[str] = []

    npz_files = sorted(run_dir.rglob("*.npz"))
    if class_filter:
        npz_files = [p for p in npz_files if class_filter in p.name.lower()]

    for npz in npz_files:
        for loader in (_load_npz_stacked, _load_npz_keyed_by_layer):
            piece = loader(npz, exclude_bos=exclude_bos)
            if piece:
                _merge(per_layer, piece)
                sources.append(f"{npz.relative_to(run_dir)} [{loader.__name__}]")
                break

    if not per_layer:
        json_files = sorted(run_dir.rglob("*.json"))
        if class_filter:
            json_files = [p for p in json_files if class_filter in p.name.lower()]
        # Heuristic: skip the per-condition aggregate.json
        json_files = [p for p in json_files if p.name != "aggregate.json"
                                            and p.name != "coverage.json"]
        for jp in json_files:
            piece = _load_response_json(jp, exclude_bos=exclude_bos)
            if piece:
                _merge(per_layer, piece)
                sources.append(f"{jp.relative_to(run_dir)} [response_json]")

    return per_layer, sources


def per_layer_coverage(per_layer: dict[int, list[np.ndarray]],
                       n_experts: int = N_ROUTED_EXPERTS) -> dict[int, dict[str, Any]]:
    """Compute unique-expert coverage per layer."""
    report: dict[int, dict[str, Any]] = {}
    for layer_idx in sorted(per_layer):
        flat_pieces = [np.asarray(a).reshape(-1) for a in per_layer[layer_idx]]
        if not flat_pieces:
            continue
        flat = np.concatenate(flat_pieces).astype(np.int64, copy=False)
        # Defensive: ignore anything outside the expert range
        valid = flat[(flat >= 0) & (flat < n_experts)]
        unique = np.unique(valid)
        report[layer_idx] = {
            "n_unique_experts": int(unique.size),
            "coverage": float(unique.size) / float(n_experts),
            "n_total_selections": int(valid.size),
            "n_invalid_selections": int(flat.size - valid.size),
        }
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run-dir", type=Path,
                    default=Path("runs/phase2_validation/routing"),
                    help="Phase 2 routing validation output directory "
                         "(default: runs/phase2_validation/routing).")
    ap.add_argument("--threshold", type=float, default=COVERAGE_THRESHOLD,
                    help=f"Per-layer coverage threshold (default: {COVERAGE_THRESHOLD}).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path (default: <run-dir>/coverage.json).")
    ap.add_argument("--no-per-class", action="store_true",
                    help="Skip per-class (toxic/benign) breakdown; aggregate only.")
    ap.add_argument("--exclude-bos", action="store_true",
                    help="Drop token index 0 from every per-prompt routing payload "
                         "before computing coverage. Diagnostic for early-MoE-layer "
                         "concentration: BOS routing is deterministic across prompts "
                         "at L3 by construction (see PHASE2_REFERENCE §7.3), so the "
                         "single token-0 selection contributes 8 fixed experts to "
                         "every prompt regardless of content. Use this to test "
                         "whether sub-50%% coverage at early layers is BOS-driven "
                         "or reflects genuine early-layer expert concentration.")
    ap.add_argument("--quiet", action="store_true", help="Suppress per-layer warnings.")
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        print(f"ERROR: run-dir does not exist: {args.run_dir}", file=sys.stderr)
        return 2

    out_path = args.out or (args.run_dir / "coverage.json")

    # Per-class load
    per_class: dict[str, dict[int, list[np.ndarray]]] = {}
    per_class_sources: dict[str, list[str]] = {}

    if not args.no_per_class:
        for cls in ("toxic", "benign"):
            data, sources = collect_class(args.run_dir, class_filter=cls,
                                          exclude_bos=args.exclude_bos)
            if data:
                per_class[cls] = data
                per_class_sources[cls] = sources

    # If nothing matched per-class, try with no filter (single-aggregate layout)
    if not per_class:
        data, sources = collect_class(args.run_dir, class_filter=None,
                                      exclude_bos=args.exclude_bos)
        if data:
            per_class["all"] = data
            per_class_sources["all"] = sources

    if not per_class:
        print(f"ERROR: no routing data found under {args.run_dir}.", file=sys.stderr)
        print("       Looked for NPZ files (stacked or per-layer-keyed) and raw "
              "response JSON files with a top-level 'routing' field.", file=sys.stderr)
        return 2

    print(f"Loaded routing data from {args.run_dir}:")
    for cls, srcs in per_class_sources.items():
        print(f"  [{cls}] {len(per_class[cls])} layers, {len(srcs)} source file(s)")
        for s in srcs[:5]:
            print(f"    - {s}")
        if len(srcs) > 5:
            print(f"    - ... +{len(srcs) - 5} more")

    # Coverage
    report: dict[str, Any] = {
        "n_routed_experts": N_ROUTED_EXPERTS,
        "threshold": args.threshold,
        "exclude_bos": bool(args.exclude_bos),
        "expected_moe_layers": list(range(FIRST_MOE_LAYER, FIRST_MOE_LAYER + N_MOE_LAYERS)),
        "per_class": {},
        "aggregate": {},
    }
    for cls, per_layer in per_class.items():
        report["per_class"][cls] = per_layer_coverage(per_layer)

    # Aggregate across classes
    merged: dict[int, list[np.ndarray]] = {}
    for per_layer in per_class.values():
        _merge(merged, per_layer)
    agg = per_layer_coverage(merged)
    report["aggregate"] = agg

    # Pass/fail on aggregate
    seen_layers = set(agg)
    expected_layers = set(report["expected_moe_layers"])
    missing_layers = sorted(expected_layers - seen_layers)
    failing_layers = sorted(li for li, r in agg.items() if r["coverage"] < args.threshold)

    report["missing_moe_layers"] = missing_layers
    report["failing_layers"] = failing_layers
    report["pass"] = (not failing_layers) and (not missing_layers)

    out_path.write_text(json.dumps(report, indent=2))

    # Console summary
    coverages = np.array([r["coverage"] for r in agg.values()])
    n_total_selections = int(sum(r["n_total_selections"] for r in agg.values()))
    print()
    bos_note = "  [BOS-token-0 excluded]" if args.exclude_bos else ""
    print(f"=== Routing coverage (threshold ≥ {args.threshold*100:.0f}% of "
          f"{N_ROUTED_EXPERTS} experts){bos_note} ===")
    print(f"Layers analyzed:       {len(agg)} (expected MoE layers: {N_MOE_LAYERS})")
    print(f"Total selections:      {n_total_selections:,}")
    print(f"Coverage min/med/max:  "
          f"{coverages.min():.3f} / {float(np.median(coverages)):.3f} / {coverages.max():.3f}")
    print(f"Layers ≥ threshold:    {int((coverages >= args.threshold).sum())}/{len(coverages)}")

    if missing_layers:
        print(f"\nWARN — {len(missing_layers)} expected MoE layer(s) absent from data:")
        print(f"  {missing_layers}")

    if failing_layers:
        print(f"\nFAIL — {len(failing_layers)} layer(s) below threshold:")
        for li in failing_layers:
            r = agg[li]
            print(f"  L{li:>2}: {r['n_unique_experts']:>3}/{N_ROUTED_EXPERTS} "
                  f"= {r['coverage']:.3f}")

    if report["pass"]:
        print(f"\nPASS — all {len(agg)} MoE layers meet ≥ {args.threshold:.2f} coverage.")

    # Per-class breakdown (compact)
    if len(per_class) > 1:
        print("\nPer-class coverage (median across layers):")
        for cls, cls_report in report["per_class"].items():
            cls_cov = np.array([r["coverage"] for r in cls_report.values()])
            print(f"  [{cls}] median={float(np.median(cls_cov)):.3f}  "
                  f"min={cls_cov.min():.3f}  max={cls_cov.max():.3f}  "
                  f"n_layers={len(cls_cov)}")

    print(f"\nWrote: {out_path}")
    return 0 if report["pass"] else 3


if __name__ == "__main__":
    sys.exit(main())
