"""
phase2_capture.py — Phase 2 capture driver.

Drives the four Phase 3 instrumentation conditions via the captures.py library,
producing per-condition outputs that the Phase 3 `vllm bench`-based harness
will reproduce.

Exit codes:
   0  success
   2  user error (bad CLI args, sidecar URL missing, etc.)
   3  value-range / non-degeneracy assertion failed
   4  zero successful calls (likely connection / auth problem)

Usage examples:

  # Phase 2 validation runs (one per condition; --n-pairs 50 is the pass-criterion size)
  python phase2_capture.py --condition baseline   --n-pairs 50
  python phase2_capture.py --condition repe_bundle --n-pairs 50
  python phase2_capture.py --condition routing     --n-pairs 50

  # 5-pair smoke for fast iteration
  python phase2_capture.py --condition repe_bundle --n-pairs 5 --dump-first-response

  # After eyeballing the entropy distribution from a smoke, lock in thresholds:
  python phase2_capture.py --condition repe_bundle --n-pairs 50 \
      --entropy-std-min 0.05 --entropy-range-min 0.2

  # Custom xargs combos (orthogonal to --condition; use either route, not both)
  python phase2_capture.py --capture-residual --capture-routing --n-pairs 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean, median, stdev
from typing import Iterable

from openai import OpenAI

from captures import (
    CaptureRequest,
    CaptureResult,
    DEFAULT_PROBE_LAYERS,
    DEFAULT_ROUTING_LAYERS,
    GLM51_FIRST_DENSE_LAYERS,
    GLM51_N_ROUTED_EXPERTS,
    GLM51_MOE_TOP_K,
    assert_activations_valid,
    assert_attention_stats_valid,
    assert_routing_valid,
    build_contrastive_pairs,
    call_with_capture,
    check_entropy_non_degenerate,
    compute_entropy_diagnostics,
    quick_diagnostics,
    save_attention_npz,
    save_residual_npz,
    save_routing_npz,
    ValueRangeError,
)


# ===== condition → xargs mapping ============================================
# This is the single source of truth that maps the four Phase 3 cells to the
# vllm_xargs combination that exercises each. Phase 3's harness reproduces
# these by reading this file (or its eventual extraction into a YAML).

CONDITION_PRESETS: dict[str, dict] = {
    "baseline": {
        "description": "No instrumentation. Phase 1 image, no vllm_xargs.",
        "residual_layers": None,
        "routing_layers": None,
        "attention_layers": None,
    },
    "repe_bundle": {
        "description": "EII-1+3+4: residual stream + per-layer attention stats on probe layers.",
        "residual_layers": DEFAULT_PROBE_LAYERS,
        "routing_layers": None,
        "attention_layers": DEFAULT_PROBE_LAYERS,
    },
    "routing": {
        "description": "MoE routing capture across all 75 MoE layers.",
        "residual_layers": None,
        "routing_layers": DEFAULT_ROUTING_LAYERS,
        "attention_layers": None,
    },
    # gradient: held until E3 lands (post-fp8-autograd-spike).
}


def build_request_from_args(args) -> CaptureRequest:
    """--condition wins if given; else compose from --capture-* flags."""
    if args.condition:
        if args.condition == "gradient":
            print(
                "[error] gradient condition is held: E3 sidecar not implemented yet.\n"
                "        Blocked on fp8-autograd spike (Plan §11 Q2). Run the other\n"
                "        three conditions first; we'll add this once the spike resolves.",
                file=sys.stderr,
            )
            sys.exit(2)
        preset = CONDITION_PRESETS[args.condition]
        return CaptureRequest(
            residual_layers=preset["residual_layers"],
            routing_layers=preset["routing_layers"],
            attention_layers=preset["attention_layers"],
        )
    # No --condition: orthogonal flag composition
    return CaptureRequest(
        residual_layers=DEFAULT_PROBE_LAYERS if args.capture_residual else None,
        routing_layers=DEFAULT_ROUTING_LAYERS if args.capture_routing else None,
        attention_layers=DEFAULT_PROBE_LAYERS if args.capture_attention else None,
    )


# ===== run loop =============================================================

def run_pass(
    client: OpenAI,
    model: str,
    prompts: list[str],
    request: CaptureRequest,
    max_new_tokens: int,
    label: str,
    out_dir: Path,
    dump_first: bool,
) -> list[CaptureResult]:
    results: list[CaptureResult] = []
    for i, p in enumerate(prompts):
        dump = (out_dir / f"raw_first_{label}.json") if (dump_first and i == 0) else None
        r = call_with_capture(client, model, p, request, max_new_tokens, dump_path=dump)
        results.append(r)
        if r.error:
            print(f"  [{label} {i+1}/{len(prompts)}] ERROR: {r.error}")
        elif (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{label} {i+1}/{len(prompts)}] "
                f"wall={r.wall_seconds:.2f}s payload~{r.payload_bytes//1024}KB"
            )
    return results


# ===== per-call assertions ==================================================

def run_per_call_assertions(
    results: list[CaptureResult],
    label: str,
    expect_residual: bool,
    expect_routing: bool,
    expect_attention: bool,
) -> list[str]:
    """Apply value-range asserts on every successful result. Returns failure messages."""
    failures: list[str] = []
    for i, r in enumerate(results):
        if r.error:
            continue
        if expect_residual:
            if r.activations is None:
                failures.append(f"[{label}#{i}] expected residual, got None")
            else:
                try:
                    assert_activations_valid(r.activations)
                except ValueRangeError as e:
                    failures.append(f"[{label}#{i}] residual range fail: {e}")
        if expect_routing:
            if r.routing is None:
                failures.append(f"[{label}#{i}] expected routing, got None")
            else:
                try:
                    assert_routing_valid(r.routing, GLM51_N_ROUTED_EXPERTS, GLM51_MOE_TOP_K)
                except ValueRangeError as e:
                    failures.append(f"[{label}#{i}] routing range fail: {e}")
        if expect_attention:
            if r.attention_stats is None:
                failures.append(f"[{label}#{i}] expected attention_stats, got None")
            else:
                try:
                    assert_attention_stats_valid(r.attention_stats)
                except ValueRangeError as e:
                    failures.append(f"[{label}#{i}] attention range fail: {e}")
    return failures


# ===== aggregate stats ======================================================

def aggregate_stats(results: list[CaptureResult]) -> dict:
    ok = [r for r in results if r.error is None]
    err = [r for r in results if r.error is not None]
    walls = [r.wall_seconds for r in ok]
    sizes = [r.payload_bytes for r in ok]

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "min": float(min(xs)),
            "median": float(median(xs)),
            "mean": float(mean(xs)),
            "max": float(max(xs)),
            "stdev": float(stdev(xs)) if len(xs) > 1 else 0.0,
        }

    return {
        "n_total": len(results),
        "n_success": len(ok),
        "n_error": len(err),
        "success_rate": len(ok) / max(1, len(results)),
        "wall_seconds": _stats(walls),
        "payload_bytes": _stats([float(s) for s in sizes]),
        "errors_sample": [r.error for r in err[:5]],
    }


# ===== main =================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--condition", choices=list(CONDITION_PRESETS.keys()) + ["gradient"],
                   default=None,
                   help="Phase 3 cell to exercise. Sets xargs; overrides --capture-*.")
    p.add_argument("--capture-residual", action="store_true",
                   help="Request residual stream at default probe layers.")
    p.add_argument("--capture-routing", action="store_true",
                   help="Request MoE routing at all 75 MoE layers.")
    p.add_argument("--capture-attention", action="store_true",
                   help="Request attention stats at default probe layers.")

    p.add_argument("--n-pairs", type=int, default=5)
    p.add_argument("--max-prompt-chars", type=int, default=600)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--model", default="zai-org/GLM-5.1-FP8")
    p.add_argument("--base-url",
                   default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
    p.add_argument("--api-key",
                   default=os.environ.get("VLLM_API_KEY", "EMPTY"))

    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: runs/phase2_validation/<condition_or_custom>/")
    p.add_argument("--dump-first-response", action="store_true",
                   help="Save raw JSON of first toxic/benign response for schema inspection.")

    # Entropy non-degeneracy thresholds (None → report-only; set after smoke).
    p.add_argument("--entropy-std-min", type=float, default=None,
                   help="Per-(layer,head) entropy std across prompts must exceed this. "
                        "None = report-only.")
    p.add_argument("--entropy-range-min", type=float, default=None,
                   help="Per-(layer,head) max-min entropy across prompts must exceed this. "
                        "None = report-only.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    request = build_request_from_args(args)
    if not request.any_capture() and args.condition != "baseline":
        print("[error] no capture requested. Pass --condition or one of --capture-*",
              file=sys.stderr)
        return 2

    label = args.condition or "custom"
    out_dir = args.out_dir or Path(f"runs/phase2_validation/{label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase2] condition={label} model={args.model} base_url={args.base_url}")
    print(f"[phase2] xargs={request.to_xargs()}")
    print(f"[phase2] out_dir={out_dir}")
    print(f"[phase2] building contrastive set: {args.n_pairs} pairs")

    pairs = build_contrastive_pairs(args.n_pairs, args.max_prompt_chars, args.seed)
    (out_dir / "pairs.json").write_text(
        json.dumps([p.__dict__ for p in pairs], indent=2)
    )

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print("[phase2] capture pass: toxic")
    toxic = run_pass(client, args.model, [p.toxic for p in pairs], request,
                     args.max_new_tokens, "toxic", out_dir, args.dump_first_response)
    print("[phase2] capture pass: benign")
    benign = run_pass(client, args.model, [p.benign for p in pairs], request,
                      args.max_new_tokens, "benign", out_dir, args.dump_first_response)

    # ---- save raw artifacts ----
    (out_dir / "generations.json").write_text(json.dumps(
        {
            "toxic":  [{"prompt": r.prompt, "text": r.text, "error": r.error} for r in toxic],
            "benign": [{"prompt": r.prompt, "text": r.text, "error": r.error} for r in benign],
        },
        indent=2,
    ))
    if request.residual_layers:
        save_residual_npz(out_dir / "toxic_residual.npz",  toxic,  request.residual_layers)
        save_residual_npz(out_dir / "benign_residual.npz", benign, request.residual_layers)
    if request.routing_layers:
        save_routing_npz(out_dir / "toxic_routing.npz",  toxic)
        save_routing_npz(out_dir / "benign_routing.npz", benign)
    if request.attention_layers:
        save_attention_npz(out_dir / "toxic_attention.npz",  toxic)
        save_attention_npz(out_dir / "benign_attention.npz", benign)

    # ---- aggregate stats ----
    agg = {
        "condition": label,
        "xargs": request.to_xargs(),
        "n_pairs": args.n_pairs,
        "model": args.model,
        "toxic":  aggregate_stats(toxic),
        "benign": aggregate_stats(benign),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print("\n[phase2] aggregate:")
    print(json.dumps(agg, indent=2))

    if agg["toxic"]["n_success"] == 0 and agg["benign"]["n_success"] == 0:
        print("[fail] 0 successful calls. Check VLLM_BASE_URL / VLLM_API_KEY.",
              file=sys.stderr)
        return 4

    # ---- baseline: verify NO instrumentation payload sneaked in ----
    if args.condition == "baseline":
        leaks: list[str] = []
        for label_, results in (("toxic", toxic), ("benign", benign)):
            for i, r in enumerate(results):
                if r.activations or r.routing or r.attention_stats:
                    leaks.append(f"{label_}#{i} unexpected payload")
        if leaks:
            print(f"[fail] baseline contaminated by instrumentation payload: "
                  f"{len(leaks)} requests (first: {leaks[0]})", file=sys.stderr)
            return 3
        print("[phase2] baseline pass: no instrumentation payload in any response.")

    # ---- per-call value-range assertions ----
    expect_residual  = request.residual_layers  is not None
    expect_routing   = request.routing_layers   is not None
    expect_attention = request.attention_layers is not None
    if expect_residual or expect_routing or expect_attention:
        fails = (
            run_per_call_assertions(toxic, "toxic",  expect_residual, expect_routing, expect_attention) +
            run_per_call_assertions(benign, "benign", expect_residual, expect_routing, expect_attention)
        )
        if fails:
            print(f"\n[fail] {len(fails)} per-call value-range failures (first 10):",
                  file=sys.stderr)
            for f in fails[:10]:
                print(f"  {f}", file=sys.stderr)
            (out_dir / "value_range_failures.json").write_text(json.dumps(fails, indent=2))
            return 3
        print("[phase2] per-call value-range checks: all PASS.")

    # ---- residual diagnostics (Phase 1 parity) ----
    if expect_residual and request.residual_layers:
        diag = quick_diagnostics(toxic, benign, request.residual_layers)
        (out_dir / "residual_diagnostics.json").write_text(json.dumps(diag, indent=2))
        print("\n[phase2] residual diagnostics (cos-sim should drop with depth):")
        print(json.dumps(diag, indent=2))

    # ---- attention entropy non-degeneracy ----
    if expect_attention:
        # Aggregate across both halves: a degenerate hook is degenerate regardless of class
        all_results = toxic + benign
        diag_ent = compute_entropy_diagnostics(all_results, aggregation="last_token")
        if diag_ent is None:
            print("[warn] no per_head_entropy data despite --capture-attention",
                  file=sys.stderr)
        else:
            (out_dir / "entropy_diagnostics.json").write_text(
                json.dumps(diag_ent.to_json(), indent=2)
            )
            report = check_entropy_non_degenerate(
                diag_ent, args.entropy_std_min, args.entropy_range_min,
            )
            (out_dir / "entropy_check.json").write_text(json.dumps(report, indent=2))
            print(f"\n[phase2] entropy non-degeneracy: {report['status']}")
            print(json.dumps(report["summary"], indent=2))
            if report["status"] == "FAIL":
                print("[fail] entropy non-degeneracy thresholds violated. "
                      "See entropy_check.json for failing (layer,head) pairs.",
                      file=sys.stderr)
                return 3

    print(f"\n[phase2] condition '{label}' complete. Artifacts in {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
