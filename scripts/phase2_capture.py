"""
phase2_capture.py — Phase 2 capture driver.

Drives the four Phase 3 instrumentation conditions via the captures.py library,
producing per-condition outputs that the Phase 3 `vllm bench`-based harness
will reproduce.

Conditions:
  baseline      — no instrumentation; vLLM deployment, no xargs.
  repe_bundle   — residual + attention_stats at probe layers; vLLM deployment.
  routing       — MoE routing at all 75 MoE layers; vLLM deployment.
  gradient      — single backward pass per request via the gradient sidecar
                  deployment (prepilot-vllm-lens-grad). EII-2 / E3.
                  Uses --grad-base-url, NOT --base-url.

Exit codes:
   0  success
   2  user error (bad CLI args, sidecar URL missing, etc.)
   3  value-range / non-degeneracy assertion failed
   4  zero successful calls (likely connection / auth problem)

Usage examples:

  # The three xargs-based conditions (vLLM deployment):
  python phase2_capture.py --condition baseline    --n-pairs 50
  python phase2_capture.py --condition repe_bundle --n-pairs 50
  python phase2_capture.py --condition routing     --n-pairs 50

  # Gradient condition (sidecar deployment):
  export VLLM_GRADIENT_BASE_URL=https://glm-5-1-prepilot-grad-<id>.tinfoil.sh
  python phase2_capture.py --condition gradient    --n-pairs 50

  # After eyeballing the gradient norm distribution from a smoke, lock thresholds:
  python phase2_capture.py --condition gradient --n-pairs 50 \
      --gradient-total-norm-std-min 1.0 --gradient-max-token-norm-std-min 0.1
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
# Gradient additions. If captures_gradients.py was merged into captures.py,
# change this import to `from captures import ...` instead.
from captures_gradients import (
    assert_gradients_valid,
    call_with_gradient_capture,
    check_gradients_non_degenerate,
    compute_gradient_diagnostics,
    save_gradients_npz,
)


# ===== condition → xargs mapping ============================================
# Single source of truth that maps the four Phase 3 cells to the vllm_xargs
# combination (or sidecar dispatch) that exercises each. Phase 3's harness
# reproduces these by reading this file (or its eventual YAML extraction).

CONDITION_PRESETS: dict[str, dict] = {
    "baseline": {
        "description": "No instrumentation. Phase 1 image, no vllm_xargs.",
        "is_sidecar": False,
        "residual_layers": None,
        "routing_layers": None,
        "attention_layers": None,
    },
    "repe_bundle": {
        "description": "EII-1+3+4: residual stream + per-layer attention stats on probe layers.",
        "is_sidecar": False,
        "residual_layers": DEFAULT_PROBE_LAYERS,
        "routing_layers": None,
        "attention_layers": DEFAULT_PROBE_LAYERS,
    },
    "routing": {
        "description": "MoE routing capture across all 75 MoE layers.",
        "is_sidecar": False,
        "residual_layers": None,
        "routing_layers": DEFAULT_ROUTING_LAYERS,
        "attention_layers": None,
    },
    "gradient": {
        "description": "EII-2: single backward pass per request → input-token gradient. "
                       "Routed to the prepilot-vllm-lens-grad sidecar deployment.",
        "is_sidecar": True,
        "residual_layers": None,
        "routing_layers": None,
        "attention_layers": None,
    },
}


def build_request_from_args(args) -> CaptureRequest:
    """--condition wins if given; else compose from --capture-* flags.

    Sidecar conditions return an empty CaptureRequest — the dispatch in main()
    routes them to the sidecar path before xargs are ever used.
    """
    if args.condition:
        preset = CONDITION_PRESETS[args.condition]
        return CaptureRequest(
            residual_layers=preset["residual_layers"],
            routing_layers=preset["routing_layers"],
            attention_layers=preset["attention_layers"],
        )
    return CaptureRequest(
        residual_layers=DEFAULT_PROBE_LAYERS if args.capture_residual else None,
        routing_layers=DEFAULT_ROUTING_LAYERS if args.capture_routing else None,
        attention_layers=DEFAULT_PROBE_LAYERS if args.capture_attention else None,
    )


# ===== run loop (vLLM/xargs path) ===========================================

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


# ===== run loop (gradient sidecar path) =====================================

def run_pass_gradient(
    base_url: str,
    api_key: str,
    prompts: list[str],
    label: str,
    out_dir: Path,
    dump_first: bool,
    timeout: float,
) -> list[CaptureResult]:
    results: list[CaptureResult] = []
    for i, p in enumerate(prompts):
        dump = (out_dir / f"raw_first_{label}.json") if (dump_first and i == 0) else None
        r = call_with_gradient_capture(
            base_url, api_key, p, timeout=timeout, dump_path=dump,
        )
        results.append(r)
        if r.error:
            print(f"  [{label} {i+1}/{len(prompts)}] ERROR: {r.error}")
            continue
        # Per-request gradient diagnostics are useful enough to print every
        # call at first; throttle later if it gets noisy. Backward latency is
        # the headline number we don't have a prior for.
        diag = r.raw.get("diagnostics", {})
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{label} {i+1}/{len(prompts)}] "
                f"wall={r.wall_seconds:.2f}s fwd={diag.get('fwd_seconds', 0):.2f}s "
                f"bwd={diag.get('bwd_seconds', 0):.2f}s "
                f"payload~{r.payload_bytes//1024}KB"
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


def run_per_call_gradient_assertions(
    results: list[CaptureResult],
    label: str,
) -> list[str]:
    """Apply value-range asserts on every successful gradient result."""
    failures: list[str] = []
    for i, r in enumerate(results):
        if r.error:
            continue
        if r.gradients is None:
            failures.append(f"[{label}#{i}] expected gradients, got None")
            continue
        try:
            assert_gradients_valid(r.gradients)
        except ValueRangeError as e:
            failures.append(f"[{label}#{i}] gradients range fail: {e}")
    return failures


# ===== aggregate stats ======================================================

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


def aggregate_stats(results: list[CaptureResult]) -> dict:
    ok = [r for r in results if r.error is None]
    err = [r for r in results if r.error is not None]
    return {
        "n_total": len(results),
        "n_success": len(ok),
        "n_error": len(err),
        "success_rate": len(ok) / max(1, len(results)),
        "wall_seconds": _stats([r.wall_seconds for r in ok]),
        "payload_bytes": _stats([float(r.payload_bytes) for r in ok]),
        "errors_sample": [r.error for r in err[:5]],
    }


def aggregate_gradient_stats(results: list[CaptureResult]) -> dict:
    """Extends aggregate_stats with gradient diagnostics from the response.

    Adds loss, fwd_seconds, bwd_seconds distributions — the operational
    quantities Phase 3 will diff against CC-on/CC-off.
    """
    base = aggregate_stats(results)
    ok = [r for r in results if r.error is None]

    def _diag_stat(key: str) -> dict:
        xs = [
            float(r.raw["diagnostics"][key])
            for r in ok
            if isinstance(r.raw, dict)
            and isinstance(r.raw.get("diagnostics"), dict)
            and key in r.raw["diagnostics"]
        ]
        return _stats(xs)

    base["loss"] = _diag_stat("loss")
    base["fwd_seconds"] = _diag_stat("fwd_seconds")
    base["bwd_seconds"] = _diag_stat("bwd_seconds")
    return base


# ===== main =================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--condition", choices=list(CONDITION_PRESETS.keys()),
                   default=None,
                   help="Phase 3 cell to exercise. Sets xargs; overrides --capture-*.")
    p.add_argument("--capture-residual", action="store_true",
                   help="Request residual stream at default probe layers.")
    p.add_argument("--capture-routing", action="store_true",
                   help="Request MoE routing at all 75 MoE layers.")
    p.add_argument("--capture-attention", action="store_true",
                   help="Request attention stats at default probe layers.")

    p.add_argument("--n-pairs", type=int, default=5)
    p.add_argument("--max-prompt-chars", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--model", default="glm-5-1-fp8")
    p.add_argument("--base-url",
                   default=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
                   help="vLLM deployment base URL (xargs-based conditions).")
    p.add_argument("--grad-base-url",
                   default=os.environ.get("VLLM_GRADIENT_BASE_URL", ""),
                   help="Gradient sidecar deployment base URL. "
                        "Required for --condition gradient.")
    p.add_argument("--api-key",
                   default=os.environ.get("VLLM_API_KEY", "EMPTY"),
                   help="Bearer token; shared between vLLM and gradient deployments.")
    p.add_argument("--grad-timeout", type=float, default=600.0,
                   help="Per-request timeout for /v1/saliency calls (seconds). "
                        "Default 600s; backward pass at frontier scale is slow.")

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

    # Gradient non-degeneracy thresholds (same report-only-then-set pattern).
    p.add_argument("--gradient-total-norm-std-min", type=float, default=None,
                   help="Per-prompt ||grad||_F std across prompts must exceed this. "
                        "None = report-only.")
    p.add_argument("--gradient-max-token-norm-std-min", type=float, default=None,
                   help="Per-prompt max-token-norm std across prompts must exceed this. "
                        "None = report-only.")
    return p.parse_args()


def _run_gradient(args) -> int:
    """Gradient condition: dispatches to the prepilot-vllm-lens-grad sidecar.

    Distinct from the xargs path: different endpoint (/v1/saliency), different
    request schema (no vllm_xargs), different deployment (--grad-base-url).
    """
    if not args.grad_base_url:
        print(
            "[error] --grad-base-url (or VLLM_GRADIENT_BASE_URL) is required for "
            "--condition gradient. The gradient sidecar is a separate Tinfoil "
            "deployment (prepilot-vllm-lens-grad); --base-url points at the vLLM "
            "deployment and is not used here.",
            file=sys.stderr,
        )
        return 2

    label = "gradient"
    out_dir = args.out_dir or Path(f"runs/phase2_validation/{label}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase2] condition={label} model={args.model} grad_base_url={args.grad_base_url}")
    print(f"[phase2] endpoint=/v1/saliency  timeout={args.grad_timeout:.0f}s")
    print(f"[phase2] out_dir={out_dir}")
    print(f"[phase2] building contrastive set: {args.n_pairs} pairs")

    pairs = build_contrastive_pairs(args.n_pairs, args.max_prompt_chars, args.seed)
    (out_dir / "pairs.json").write_text(
        json.dumps([p.__dict__ for p in pairs], indent=2)
    )

    print("[phase2] capture pass: toxic")
    toxic = run_pass_gradient(
        args.grad_base_url, args.api_key,
        [p.toxic for p in pairs], "toxic", out_dir,
        args.dump_first_response, args.grad_timeout,
    )
    print("[phase2] capture pass: benign")
    benign = run_pass_gradient(
        args.grad_base_url, args.api_key,
        [p.benign for p in pairs], "benign", out_dir,
        args.dump_first_response, args.grad_timeout,
    )

    # ---- save raw artifacts ----
    # generations.json: for the gradient flow, `text` is the argmax target token,
    # not a completion. Keep the key for schema consistency with other conditions
    # but rename to target_token in the JSON to avoid downstream confusion.
    (out_dir / "generations.json").write_text(json.dumps(
        {
            "toxic": [{"prompt": r.prompt, "target_token": r.text,
                       "prompt_tokens": r.prompt_tokens, "error": r.error} for r in toxic],
            "benign": [{"prompt": r.prompt, "target_token": r.text,
                        "prompt_tokens": r.prompt_tokens, "error": r.error} for r in benign],
        },
        indent=2,
    ))
    save_gradients_npz(out_dir / "toxic_gradients.npz",  toxic)
    save_gradients_npz(out_dir / "benign_gradients.npz", benign)

    # ---- aggregate stats ----
    agg = {
        "condition": label,
        "xargs": {},
        "n_pairs": args.n_pairs,
        "model": args.model,
        "endpoint": "/v1/saliency",
        "deployment": "prepilot-vllm-lens-grad",
        "toxic":  aggregate_gradient_stats(toxic),
        "benign": aggregate_gradient_stats(benign),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2))
    print("\n[phase2] aggregate:")
    print(json.dumps(agg, indent=2))

    if agg["toxic"]["n_success"] == 0 and agg["benign"]["n_success"] == 0:
        print(
            "[fail] 0 successful calls. Check --grad-base-url / --api-key / sidecar status.",
            file=sys.stderr,
        )
        return 4

    # ---- per-call value-range assertions ----
    fails = (
        run_per_call_gradient_assertions(toxic, "toxic")
        + run_per_call_gradient_assertions(benign, "benign")
    )
    if fails:
        print(f"\n[fail] {len(fails)} per-call value-range failures (first 10):",
              file=sys.stderr)
        for f in fails[:10]:
            print(f"  {f}", file=sys.stderr)
        (out_dir / "value_range_failures.json").write_text(json.dumps(fails, indent=2))
        return 3
    print("[phase2] per-call value-range checks: all PASS.")

    # ---- cross-prompt non-degeneracy ----
    all_results = toxic + benign
    diag = compute_gradient_diagnostics(all_results)
    if diag is None:
        print("[warn] no gradient data despite gradient condition", file=sys.stderr)
    else:
        (out_dir / "gradient_diagnostics.json").write_text(
            json.dumps(diag.to_json(), indent=2)
        )
        report = check_gradients_non_degenerate(
            diag,
            total_norm_std_min=args.gradient_total_norm_std_min,
            max_token_norm_std_min=args.gradient_max_token_norm_std_min,
        )
        (out_dir / "gradient_check.json").write_text(json.dumps(report, indent=2))
        print(f"\n[phase2] gradient non-degeneracy: {report['status']}")
        print(json.dumps(report["summary"], indent=2))
        if report["status"] == "FAIL":
            print(
                "[fail] gradient non-degeneracy thresholds violated. "
                "See gradient_check.json.",
                file=sys.stderr,
            )
            return 3

    print(f"\n[phase2] condition '{label}' complete. Artifacts in {out_dir}/")
    return 0


def main() -> int:
    args = parse_args()

    # Dispatch the sidecar (gradient) condition before touching the OpenAI client.
    if args.condition == "gradient":
        return _run_gradient(args)

    # ---- xargs-based path (baseline / repe_bundle / routing / custom) ----
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
            "toxic": [{"prompt": r.prompt, "text": r.text,
              "prompt_tokens": r.prompt_tokens, "error": r.error} for r in toxic],
            "benign": [{"prompt": r.prompt, "text": r.text,
              "prompt_tokens": r.prompt_tokens, "error": r.error} for r in benign],
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
