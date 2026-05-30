#!/usr/bin/env python3
"""
build_pairs.py — one-shot generator for the fork's data/ directory.

Builds the deterministic 50-toxic + 50-benign ToxicChat prompt set used by:
  - phase3_grad_driver.py / phase3_vllm_driver.py  (reads pairs.json)
  - phase3_run_matrix.py vllm bench reference run  (reads pairs.jsonl)

Both files are produced from the same call to captures.build_contrastive_pairs
with identical (seed, max_prompt_chars) so cross-tool comparability is
guaranteed.

Run once during fork setup; commit the outputs to data/ in the benchbox fork.
The benchbox Dockerfile then COPYs data/ into /workspace/data/ at image build
time, so the deployed container is self-contained — no HF download at runtime.

Usage:

    python scripts/build_pairs.py --out-dir data --n-pairs 50

Output:

    data/pairs.json   — list of {pair_id, toxic, benign} (drivers' input format)
    data/pairs.jsonl  — one {"prompt": str} per line, toxic then benign per pair
                        (vllm bench --dataset-name custom format)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# captures.py is expected to sit alongside this script under scripts/.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from captures import build_contrastive_pairs  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=Path("data"))
    p.add_argument("--n-pairs", type=int, default=50,
                   help="Number of toxic-benign pairs. 50 matches Phase 2 corpus exactly.")
    p.add_argument("--max-prompt-chars", type=int, default=1024,
                   help="Char-level filter on prompts. 1024 matches Phase 2.")
    p.add_argument("--seed", type=int, default=0,
                   help="Shuffle seed. 0 matches Phase 2.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[build] n_pairs={args.n_pairs} max_prompt_chars={args.max_prompt_chars} "
          f"seed={args.seed}")
    pairs = build_contrastive_pairs(args.n_pairs, args.max_prompt_chars, args.seed)

    # pairs.json — drivers' format
    pairs_json_path = args.out_dir / "pairs.json"
    pairs_json_path.write_text(json.dumps(
        [{"pair_id": p.pair_id, "toxic": p.toxic, "benign": p.benign} for p in pairs],
        indent=2,
    ))
    print(f"[build] wrote {pairs_json_path} ({len(pairs)} pairs)")

    # pairs.jsonl — vllm bench --dataset-name custom format
    # Ordering matches phase3 driver interleave: toxic[0], benign[0], toxic[1], ...
    pairs_jsonl_path = args.out_dir / "pairs.jsonl"
    with pairs_jsonl_path.open("w") as f:
        for p in pairs:
            f.write(json.dumps({"prompt": p.toxic}) + "\n")
            f.write(json.dumps({"prompt": p.benign}) + "\n")
    print(f"[build] wrote {pairs_jsonl_path} ({2 * len(pairs)} prompts)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
