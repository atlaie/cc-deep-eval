"""
sanity_check.py — Phase 1 reproduction driver.

Refactored 2026-05-12 to share code with captures.py and adopt RepE-canonical
position semantics (last formatted-prompt token, t[prompt_tokens - 1]) rather
than t[-1] on the full captured sequence. Aligns Phase 1 with Phase 2 so
cos_sim numbers are directly comparable across phases.

Original Phase 1 numbers in PHASE1_REFERENCE.md §8 were generated under the
prior t[-1] convention against the v0.0.3 server schema. They remain in git
history; re-run this script to generate canonical Phase 1 numbers under the
unified methodology.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from captures import (
    CaptureRequest,
    DEFAULT_PROBE_LAYERS,
    build_contrastive_pairs,
    call_with_capture,
    quick_diagnostics,
    save_residual_npz,
)


DEFAULT_BASE_URL = "http://localhost:8001/v1"
DEFAULT_MODEL = "glm-5-1-fp8"
DEFAULT_OUT_DIR = Path("./runs/phase1_sanity")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default=os.environ.get("VLLM_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-pairs", type=int, default=50)
    p.add_argument("--layers", type=int, nargs="+", default=DEFAULT_PROBE_LAYERS)
    p.add_argument("--max-prompt-chars", type=int, default=1024)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dump-first-response", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase1] model={args.model} base_url={args.base_url}")
    print(f"[phase1] probing layers: {args.layers}")
    print(f"[phase1] building contrastive set: {args.n_pairs} pairs "
          f"(max_prompt_chars={args.max_prompt_chars}, seed={args.seed})")

    pairs = build_contrastive_pairs(args.n_pairs, args.max_prompt_chars, args.seed)
    (args.out_dir / "pairs.json").write_text(
        json.dumps([p.__dict__ for p in pairs], indent=2)
    )

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    request = CaptureRequest(residual_layers=args.layers)

    def _do_pass(prompts: list[str], label: str):
        results = []
        for i, p in enumerate(prompts):
            dump = (args.out_dir / f"raw_first_{label}.json"
                    if args.dump_first_response and i == 0 else None)
            r = call_with_capture(client, args.model, p, request,
                                  args.max_new_tokens, dump_path=dump)
            results.append(r)
            if r.error:
                print(f"  [{label} {i+1}/{len(prompts)}] ERROR: {r.error}")
            elif (i + 1) % 10 == 0 or i == 0:
                print(f"  [{label} {i+1}/{len(prompts)}] "
                      f"wall={r.wall_seconds:.2f}s payload~{r.payload_bytes//1024}KB")
        return results

    print("[phase1] capture pass: toxic")
    toxic = _do_pass([p.toxic for p in pairs], "toxic")
    print("[phase1] capture pass: benign")
    benign = _do_pass([p.benign for p in pairs], "benign")

    save_residual_npz(args.out_dir / "toxic.npz", toxic, args.layers)
    save_residual_npz(args.out_dir / "benign.npz", benign, args.layers)
    (args.out_dir / "generations.json").write_text(json.dumps({
        "toxic":  [{"prompt": r.prompt, "text": r.text,
                    "prompt_tokens": r.prompt_tokens, "error": r.error} for r in toxic],
        "benign": [{"prompt": r.prompt, "text": r.text,
                    "prompt_tokens": r.prompt_tokens, "error": r.error} for r in benign],
    }, indent=2))

    print("[phase1] diagnostics")
    summary = quick_diagnostics(toxic, benign, args.layers)
    (args.out_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    n_ok = sum(1 for r in toxic if r.activations)
    print(f"\n[phase1] activations captured: {n_ok}/{len(toxic)} toxic prompts")
    if not summary:
        print("\n[!!] No activations extracted. Re-run with --dump-first-response "
              "to inspect raw_first_toxic.json against captures.extract_activations.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())