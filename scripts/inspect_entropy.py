"""
inspect_entropy.py — eyeball a saved entropy_diagnostics.json and propose
threshold values for `--entropy-std-min` / `--entropy-range-min`.

Workflow:
  1. python phase2_capture.py --capture-attention --n-pairs 5
       → produces runs/.../entropy_diagnostics.json (no thresholds applied)
  2. python inspect_entropy.py runs/phase2_validation/<dir>/entropy_diagnostics.json
       → prints per-layer summary + proposed thresholds based on the actual
         distribution
  3. python phase2_capture.py --condition repe_bundle --n-pairs 50 \
         --entropy-std-min <X> --entropy-range-min <Y>

The proposed thresholds are "1st-percentile minus a safety margin" of the
observed (std, range) distribution. This is a starting point, not a verdict
— if the smoke distribution looks suspicious (e.g. half the heads have std≈0),
do NOT lock thresholds; debug the hook first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("path", type=Path,
                   help="Path to entropy_diagnostics.json from phase2_capture.py")
    p.add_argument("--safety-fraction", type=float, default=0.5,
                   help="Multiplier on the 1st-percentile observed value. "
                        "0.5 = propose threshold at half the smallest reasonable "
                        "value (default; conservative).")
    args = p.parse_args()

    if not args.path.exists():
        print(f"[error] {args.path} does not exist", file=sys.stderr)
        return 2
    d = json.loads(args.path.read_text())

    layers = np.asarray(d["layers"])
    e_mean = np.asarray(d["entropy_mean"])  # (n_layers, n_heads)
    e_std  = np.asarray(d["entropy_std"])
    e_min  = np.asarray(d["entropy_min"])
    e_max  = np.asarray(d["entropy_max"])
    e_rng  = e_max - e_min
    n_prompts = d.get("n_prompts", "?")

    print(f"=== entropy diagnostics summary ===")
    print(f"  source:      {args.path}")
    print(f"  n_prompts:   {n_prompts}")
    print(f"  aggregation: {d.get('aggregation')}")
    print(f"  layers:      {layers.tolist()}")
    print(f"  shape:       {e_mean.shape}  (n_layers, n_heads)")
    print()

    # Per-layer collapse for readability
    print(f"{'layer':>6}  {'mean(E)':>8}  {'min/median/max(std)':>26}  "
          f"{'min/median/max(range)':>26}")
    for i, L in enumerate(layers):
        print(
            f"{int(L):>6}  {e_mean[i].mean():>8.3f}  "
            f"{e_std[i].min():>7.3f}/{np.median(e_std[i]):>7.3f}/{e_std[i].max():>7.3f}"
            f"  "
            f"{e_rng[i].min():>7.3f}/{np.median(e_rng[i]):>7.3f}/{e_rng[i].max():>7.3f}"
        )

    print()
    print(f"=== distribution-wide ===")
    print(f"  std       : min={e_std.min():.4f}  "
          f"p1={np.percentile(e_std, 1):.4f}  "
          f"median={np.median(e_std):.4f}  "
          f"max={e_std.max():.4f}")
    print(f"  range     : min={e_rng.min():.4f}  "
          f"p1={np.percentile(e_rng, 1):.4f}  "
          f"median={np.median(e_rng):.4f}  "
          f"max={e_rng.max():.4f}")

    # Sanity flags
    flags: list[str] = []
    n_total = e_std.size
    n_zero_std = int((e_std < 1e-6).sum())
    n_near_log = int((e_mean > 0.9 * np.log(max(2, n_prompts if isinstance(n_prompts, int) else 32))).sum())
    if n_zero_std > 0:
        flags.append(f"  ⚠  {n_zero_std}/{n_total} heads have std<1e-6 (constant across prompts)")
    if e_std.mean() < 1e-3:
        flags.append(f"  ⚠  mean std={e_std.mean():.6f} — distribution looks degenerate")
    if (e_mean > 5.0).any():
        flags.append(f"  ⚠  some entropy means >5.0; check seq_len and log base")
    if flags:
        print(f"\n=== sanity flags ===")
        for f in flags:
            print(f)
        print(f"  → eyeball entropy_diagnostics.json before locking thresholds.")

    # Proposed thresholds: 1st-percentile × safety_fraction
    # The intuition: if the smallest 1% of heads (by std or range) already varies
    # by amount X, then a threshold at X * 0.5 will catch a genuinely broken
    # hook (std≈0) while passing this distribution.
    std_p1 = float(np.percentile(e_std, 1))
    rng_p1 = float(np.percentile(e_rng, 1))
    std_threshold = std_p1 * args.safety_fraction
    rng_threshold = rng_p1 * args.safety_fraction

    print(f"\n=== proposed thresholds (1st-pct × {args.safety_fraction}) ===")
    print(f"  --entropy-std-min   {std_threshold:.4f}")
    print(f"  --entropy-range-min {rng_threshold:.4f}")
    if flags:
        print(f"  (NOTE: sanity flags above — debug before using these.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
